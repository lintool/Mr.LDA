package cc.mrlda.polylda;

import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import cc.mrlda.InformedPrior;
import cc.mrlda.polylda.VariationalInference.ParameterCounter;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.triple.TripleOfInts;
import edu.umd.cloud9.math.Gamma;
import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.HMapIV;

public class DocumentMapper extends MapReduceBase implements
    Mapper<IntWritable, Document, TripleOfInts, DoubleWritable> {
  private long configurationTime = 0;
  private long trainingTime = 0;

  private static HMapIV<double[]>[] beta = null;

  private static double[] alpha = null;
  private static double likelihoodAlpha = 0;

  private static int numberOfTopics = 0;
  private static int numberOfLanguages = 0;
  private static int[] numberOfTerms = null;

  private static int maximumGammaIteration = Settings.MAXIMUM_LOCAL_ITERATION;

  private static boolean learning = Settings.LEARNING_MODE;
  private static boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

  private TripleOfInts outputKey = new TripleOfInts();
  private DoubleWritable outputValue = new DoubleWritable();

  // boolean seededGamma = false;
  // boolean seededAlpha = false;

  private static MultipleOutputs multipleOutputs;
  private static OutputCollector<IntWritable, Document> outputDocument;

  private double[] tempBeta = null;

  private double[] tempGamma = null;
  private double[] updateGamma = null;

  private double[] phi = null;
  private HashMap<Integer, double[]>[] phiTable = null;

  private Iterator<Integer> itr = null;
  private HMapII hmap = null;

  public void configure(JobConf conf) {
    configurationTime = System.currentTimeMillis();

    numberOfTopics = conf.getInt(Settings.PROPERTY_PREFIX + "model.topics", 0);
    numberOfLanguages = conf.getInt(Settings.PROPERTY_PREFIX + "model.languages", 0);
    beta = new HMapIV[numberOfLanguages];

    numberOfTerms = new int[numberOfLanguages];
    // language index starts from 1
    for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
      numberOfTerms[languageIndex] = conf.getInt(Settings.PROPERTY_PREFIX + "corpus.terms"
          + Settings.DOT + (languageIndex + 1), Integer.MAX_VALUE);
    }
    maximumGammaIteration = conf.getInt(Settings.PROPERTY_PREFIX
        + "model.mapper.converge.iteration", Settings.MAXIMUM_LOCAL_ITERATION);

    learning = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.train", Settings.LEARNING_MODE);
    randomStartGamma = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.random.start",
        Settings.RANDOM_START_GAMMA);

    multipleOutputs = new MultipleOutputs(conf);

    updateGamma = new double[numberOfTopics];
    phiTable = new HashMap[numberOfLanguages];
    for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
      phiTable[languageIndex] = new HashMap<Integer, double[]>();
    }

    double alphaSum = 0;

    Path[] inputFiles;
    SequenceFile.Reader sequenceFileReader = null;
    try {
      inputFiles = DistributedCache.getLocalCacheFiles(conf);
      if (inputFiles != null) {
        for (Path path : inputFiles) {
          try {
            sequenceFileReader = new SequenceFile.Reader(FileSystem.getLocal(conf), path, conf);

            if (path.getName().startsWith(Settings.BETA)) {
              // Settings.BETA + Settings.LEFT_BRACKET + languageIndex + Settings.RIGHT_BRACKET
              String fileName = path.getName();
              int languageIndex = Integer.parseInt(fileName.substring(
                  fileName.indexOf(Settings.LANGUAGE_INDICATOR)
                      + Settings.LANGUAGE_INDICATOR.length(), fileName.indexOf(Settings.DASH)));

              // TODO: check whether seeded beta is valid, i.e., a true probability distribution
              // language index starts from 1
              Preconditions.checkArgument(beta[languageIndex - 1] == null,
                  "Beta matrix was initialized already...");
              beta[languageIndex - 1] = cc.mrlda.DocumentMapper.importBeta(sequenceFileReader,
                  numberOfTopics, numberOfTerms[languageIndex - 1]);

              // System.out.println(beta[languageIndex - 1] + " ");
            } else if (path.getName().startsWith(Settings.ALPHA)) {
              Preconditions.checkArgument(alpha == null, "Alpha vector was initialized already...");

              // TODO: check the validity of alpha
              alpha = cc.mrlda.VariationalInference.importAlpha(sequenceFileReader, numberOfTopics);
              double sumLnGammaAlpha = 0;
              for (double value : alpha) {
                sumLnGammaAlpha += Gamma.lngamma(value);
                alphaSum += value;
              }
              likelihoodAlpha = Gamma.lngamma(alphaSum) - sumLnGammaAlpha;
            } else if (path.getName().startsWith(InformedPrior.ETA)) {
              // beta = parseEta(sequenceFileReader, numberOfTopics);
              continue;
            } else {
              throw new IllegalArgumentException("Unexpected file in distributed cache: "
                  + path.getName());
            }
          } catch (IllegalArgumentException iae) {
            iae.printStackTrace();
          } catch (IOException ioe) {
            ioe.printStackTrace();
          } finally {
            IOUtils.closeStream(sequenceFileReader);
          }
        }
      }
    } catch (IOException ioe) {
      ioe.printStackTrace();
    }

    for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
      if (beta[languageIndex] == null) {
        beta[languageIndex] = new HMapIV<double[]>();
      }
    }

    if (alpha == null) {
      alpha = new double[numberOfTopics];
      double alphaLnGammaSum = 0;
      for (int i = 0; i < numberOfTopics; i++) {
        alpha[i] = Math.random();
        alphaSum += alpha[i];
        alphaLnGammaSum += Gamma.lngamma(alpha[i]);
      }
      likelihoodAlpha = Gamma.lngamma(alphaSum) - alphaLnGammaSum;
    }

    // System.out.println("======================================================================");
    // System.out.println("Available processors (cores): "
    // + Runtime.getRuntime().availableProcessors());
    // long maxMemory = Runtime.getRuntime().maxMemory();
    // System.out.println("Maximum memory (bytes): "
    // + (maxMemory == Long.MAX_VALUE ? "no limit" : maxMemory));
    // System.out.println("Free memory (bytes): " + Runtime.getRuntime().freeMemory());
    // System.out.println("Total memory (bytes): " + Runtime.getRuntime().totalMemory());
    // System.out.println("======================================================================");

    configurationTime = System.currentTimeMillis() - configurationTime;
  }

  @SuppressWarnings("deprecation")
  public void map(IntWritable key, Document value,
      OutputCollector<TripleOfInts, DoubleWritable> output, Reporter reporter) throws IOException {
    reporter.incrCounter(ParameterCounter.CONFIG_TIME, configurationTime);
    reporter.incrCounter(ParameterCounter.TOTAL_DOC, 1);
    trainingTime = System.currentTimeMillis();

    double likelihoodPhi = 0;

    // initialize tempGamma for computing
    if (value.getGamma() != null && value.getNumberOfTopics() == numberOfTopics
        && !randomStartGamma) {
      // TODO: set up mechanisms to prevent starting from some irrelevant gamma value
      tempGamma = value.getGamma();
    } else {
      int totalNumberOfWords = 0;
      for (int i : value.getNumberOfWords()) {
        totalNumberOfWords += i;
      }

      tempGamma = new double[numberOfTopics];
      for (int i = 0; i < numberOfTopics; i++) {
        tempGamma[i] = alpha[i] + 1.0f * totalNumberOfWords / numberOfTopics;
      }
    }

    double[] phi = null;

    int gammaUpdateIterationCount = 1;
    do {
      likelihoodPhi = 0;

      for (int i = 0; i < numberOfTopics; i++) {
        tempGamma[i] = Gamma.digamma(tempGamma[i]);
        updateGamma[i] = Math.log(alpha[i]);
      }

      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        hmap = value.getContent(languageIndex);
        if (hmap == null) {
          continue;
        }

        itr = value.getContent(languageIndex).keySet().iterator();
        while (itr.hasNext()) {
          int termID = itr.next();
          int termCounts = hmap.get(termID);

          if (phiTable[languageIndex].containsKey(termID)) {
            // reuse existing object
            phi = phiTable[languageIndex].get(termID);
          } else {
            phi = new double[numberOfTopics];
            phiTable[languageIndex].put(termID, phi);
          }

          // acquire the corresponding beta vector for this term
          tempBeta = cc.mrlda.DocumentMapper.retrieveBeta(numberOfTopics, beta[languageIndex],
              termID, numberOfTerms[languageIndex]);

          likelihoodPhi += cc.mrlda.DocumentMapper.updatePhi(numberOfTopics, termCounts, tempBeta,
              tempGamma, phi, updateGamma);
        }
      }

      // send out heart beat message
      if (Math.random() < 0.01) {
        reporter.incrCounter(ParameterCounter.DUMMY_COUNTER, 1);
      }

      for (int i = 0; i < numberOfTopics; i++) {
        tempGamma[i] = Math.exp(updateGamma[i]);
      }

      gammaUpdateIterationCount++;
    } while (gammaUpdateIterationCount < maximumGammaIteration);

    // compute the sum of gamma vector
    double sumGamma = 0;
    double likelihoodGamma = 0;
    for (int i = 0; i < numberOfTopics; i++) {
      sumGamma += tempGamma[i];
      likelihoodGamma += Gamma.lngamma(tempGamma[i]);
    }
    likelihoodGamma -= Gamma.lngamma(sumGamma);
    double documentLogLikelihood = likelihoodAlpha + likelihoodGamma + likelihoodPhi;
    reporter.incrCounter(ParameterCounter.LOG_LIKELIHOOD,
        (long) (-documentLogLikelihood * Settings.DEFAULT_COUNTER_SCALE));

    double digammaSumGamma = Gamma.digamma(sumGamma);

    if (learning) {
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        hmap = value.getContent(languageIndex);
        if (hmap == null) {
          continue;
        }

        // emit the phi counts for each languages
        itr = hmap.keySet().iterator();
        while (itr.hasNext()) {
          int termID = itr.next();
          phi = phiTable[languageIndex].get(termID);
          for (int i = 0; i < numberOfTopics; i++) {
            // emit phi values.
            outputKey.set(languageIndex + 1, i + 1, termID);
            outputValue.set(phi[i]);
            // System.out.println("phi\t" + outputKey + "\t" + outputValue);
            output.collect(outputKey, outputValue);
          }
        }
      }

      // emit the alpha sufficient statistics
      for (int i = 0; i < numberOfTopics; i++) {
        outputKey.set(0, i + 1, 0);
        outputValue.set(Gamma.digamma(updateGamma[i]) - digammaSumGamma);
        // System.out.println("alpha\t" + outputKey + "\t" + outputValue);
        output.collect(outputKey, outputValue);
      }
    }

    // output the embedded updated gamma together with document
    if (!learning || !randomStartGamma) {
      outputDocument = multipleOutputs.getCollector(Settings.GAMMA, Settings.GAMMA, reporter);
      value.setGamma(tempGamma);
      outputDocument.collect(key, value);
    }

    trainingTime = System.currentTimeMillis() - trainingTime;
    reporter.incrCounter(ParameterCounter.TRAINING_TIME, trainingTime);
  }

  public void close() throws IOException {
    multipleOutputs.close();
  }
}