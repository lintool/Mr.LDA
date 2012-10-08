package cc.mrlda;

import java.io.IOException;
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

import cc.mrlda.VariationalInference.ParameterCounter;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapIDW;
import edu.umd.cloud9.io.pair.PairOfIntFloat;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.math.Gamma;
import edu.umd.cloud9.math.LogMath;
import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.HMapIV;

public class DocumentMapper extends MapReduceBase implements
    Mapper<IntWritable, Document, PairOfInts, DoubleWritable> {

  boolean directEmit = false;
  HMapIV<double[]> totalPhi = null;
  double[] totalAlphaSufficientStatistics;
  OutputCollector<PairOfInts, DoubleWritable> outputCollector;

  private long configurationTime = 0;
  private long trainingTime = 0;

  private static HMapIV<double[]> expectLogBeta = null;
  private static double[] alpha = null;

  private static int numberOfTopics = 0;
  private static int numberOfTerms = Integer.MAX_VALUE;

  private static int maximumGammaIteration = Settings.MAXIMUM_LOCAL_ITERATION;

  private static boolean learning = Settings.LEARNING_MODE;
  private static boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

  private static double likelihoodAlpha = 0;

  private PairOfInts outputKey = new PairOfInts();
  private DoubleWritable outputValue = new DoubleWritable();

  private static MultipleOutputs multipleOutputs;
  private static OutputCollector<IntWritable, Document> outputDocument;

  private double[] tempLogBeta = null;

  private double[] tempGamma = null;
  private double[] updateLogGamma = null;

  private HMapIV<double[]> logPhiTable = null;

  private Iterator<Integer> itr = null;

  public void configure(JobConf conf) {
    configurationTime = System.currentTimeMillis();

    numberOfTerms = conf.getInt(Settings.PROPERTY_PREFIX + "corpus.terms", Integer.MAX_VALUE);
    numberOfTopics = conf.getInt(Settings.PROPERTY_PREFIX + "model.topics", 0);
    // Settings.DEFAULT_NUMBER_OF_TOPICS);
    maximumGammaIteration = conf.getInt(Settings.PROPERTY_PREFIX
        + "model.mapper.converge.iteration", Settings.MAXIMUM_LOCAL_ITERATION);

    learning = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.train", Settings.LEARNING_MODE);
    randomStartGamma = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.random.start",
        Settings.RANDOM_START_GAMMA);

    // approximateBeta = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.truncate.beta", false);

    directEmit = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.mapper.direct.emit",
        Settings.DEFAULT_DIRECT_EMIT);
    if (!directEmit) {
      totalPhi = new HMapIV<double[]>();
    }

    totalAlphaSufficientStatistics = new double[numberOfTopics];

    updateLogGamma = new double[numberOfTopics];
    logPhiTable = new HMapIV<double[]>();

    multipleOutputs = new MultipleOutputs(conf);

    double alphaSum = 0;

    SequenceFile.Reader sequenceFileReader = null;
    try {
      Path[] inputFiles = DistributedCache.getLocalCacheFiles(conf);
      // TODO: check for the missing columns...
      if (inputFiles != null) {
        for (Path path : inputFiles) {
          try {
            sequenceFileReader = new SequenceFile.Reader(FileSystem.getLocal(conf), path, conf);

            if (path.getName().startsWith(Settings.BETA)) {
              // TODO: check whether seeded beta is valid, i.e., a true probability distribution
              Preconditions.checkArgument(expectLogBeta == null,
                  "Beta matrix was initialized already...");
              // beta = importBeta(sequenceFileReader, numberOfTopics, numberOfTerms,
              // approximateBeta);
              expectLogBeta = importBeta(sequenceFileReader, numberOfTopics, numberOfTerms);
            } else if (path.getName().startsWith(Settings.ALPHA)) {
              Preconditions.checkArgument(alpha == null, "Alpha vector was initialized already...");
              // TODO: check the validity of alpha
              alpha = VariationalInference.importAlpha(sequenceFileReader, numberOfTopics);
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

    if (expectLogBeta == null) {
      expectLogBeta = new HMapIV<double[]>();
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

    System.out.println("======================================================================");
    System.out.println("Available processors (cores): "
        + Runtime.getRuntime().availableProcessors());
    long maxMemory = Runtime.getRuntime().maxMemory();
    System.out.println("Maximum memory (bytes): "
        + (maxMemory == Long.MAX_VALUE ? "no limit" : maxMemory));
    System.out.println("Free memory (bytes): " + Runtime.getRuntime().freeMemory());
    System.out.println("Total memory (bytes): " + Runtime.getRuntime().totalMemory());
    System.out.println("======================================================================");

    configurationTime = System.currentTimeMillis() - configurationTime;
  }

  @SuppressWarnings("deprecation")
  public void map(IntWritable key, Document value,
      OutputCollector<PairOfInts, DoubleWritable> output, Reporter reporter) throws IOException {
    reporter.incrCounter(ParameterCounter.CONFIG_TIME, configurationTime);
    reporter.incrCounter(ParameterCounter.TOTAL_DOCS, 1);
    trainingTime = System.currentTimeMillis();

    double likelihoodPhi = 0;

    // initialize tempGamma for computing
    if (value.getGamma() != null && value.getNumberOfTopics() == numberOfTopics
        && !randomStartGamma) {
      // TODO: set up mechanisms to prevent starting from some irrelevant gamma value
      tempGamma = value.getGamma();
    } else {
      tempGamma = new double[numberOfTopics];
      for (int i = 0; i < numberOfTopics; i++) {
        tempGamma[i] = alpha[i] + 1.0f * value.getNumberOfTokens() / numberOfTopics;
      }
    }

    double[] logPhi = null;

    HMapII content = value.getContent();
    if (content == null) {
      System.err.println("Error - content was null for document " + key.toString());
      return;
    }

    // be careful when adjust this initial value
    int gammaUpdateIterationCount = 1;
    do {
      likelihoodPhi = 0;

      for (int i = 0; i < numberOfTopics; i++) {
        tempGamma[i] = Gamma.digamma(tempGamma[i]);
        updateLogGamma[i] = Math.log(alpha[i]);
      }

      itr = content.keySet().iterator();
      while (itr.hasNext()) {
        int termID = itr.next();
        // acquire the corresponding beta vector for this term
        if (logPhiTable.containsKey(termID)) {
          // reuse existing object
          logPhi = logPhiTable.get(termID);
        } else {
          logPhi = new double[numberOfTopics];
          logPhiTable.put(termID, logPhi);
        }

        int termCounts = content.get(termID);
        tempLogBeta = retrieveBeta(numberOfTopics, expectLogBeta, termID, numberOfTerms);

        likelihoodPhi += updatePhi(numberOfTopics, termCounts, tempLogBeta, tempGamma, logPhi,
            updateLogGamma);
      }

      for (int i = 0; i < numberOfTopics; i++) {
        tempGamma[i] = Math.exp(updateLogGamma[i]);
      }

      gammaUpdateIterationCount++;

      // send out heart-beat message
      if (Math.random() < 0.01) {
        reporter.incrCounter(ParameterCounter.DUMMY_COUNTER, 1);
      }
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
    for (int i = 0; i < numberOfTopics; i++) {
      totalAlphaSufficientStatistics[i] += Gamma.digamma(tempGamma[i]) - digammaSumGamma;
    }

    outputCollector = output;

    if (!directEmit) {
      if (learning) {
        if (Runtime.getRuntime().freeMemory() < Settings.MEMORY_THRESHOLD) {
          itr = totalPhi.keySet().iterator();
          while (itr.hasNext()) {
            int termID = itr.next();
            logPhi = totalPhi.get(termID);
            for (int i = 0; i < numberOfTopics; i++) {
              outputValue.set(logPhi[i]);

              // a *positive* topic index indicates the output is a phi values
              outputKey.set(i + 1, termID);
              output.collect(outputKey, outputValue);
            }
          }
          totalPhi.clear();

          // for (int i = 0; i < numberOfTopics; i++) {
          // a *zero* topic index and a *positive* topic index indicates the output is a term for
          // alpha updating
          // outputKey.set(0, i + 1);
          // outputValue.set(totalAlphaSufficientStatistics[i]);
          // output.collect(outputKey, outputValue);
          // totalAlphaSufficientStatistics[i] = 0;
          // }
        }

        itr = content.keySet().iterator();
        while (itr.hasNext()) {
          int termID = itr.next();
          if (termID < Settings.TOP_WORDS_FOR_CACHING) {
            if (totalPhi.containsKey(termID)) {
              logPhi = logPhiTable.get(termID);
              tempLogBeta = totalPhi.get(termID);
              for (int i = 0; i < numberOfTopics; i++) {
                tempLogBeta[i] = LogMath.add(logPhi[i], tempLogBeta[i]);
              }
            } else {
              totalPhi.put(termID, logPhiTable.get(termID));
            }
          } else {
            logPhi = logPhiTable.get(termID);
            for (int i = 0; i < numberOfTopics; i++) {
              outputValue.set(logPhi[i]);

              // a *positive* topic index indicates the output is a phi values
              outputKey.set(i + 1, termID);
              output.collect(outputKey, outputValue);
            }
          }
        }
      }
    } else {
      if (learning) {
        itr = content.keySet().iterator();
        while (itr.hasNext()) {
          int termID = itr.next();
          // only get the phi's in of current document
          logPhi = logPhiTable.get(termID);
          for (int i = 0; i < numberOfTopics; i++) {
            outputValue.set(logPhi[i]);

            // a *positive* topic index indicates the output is a phi values
            outputKey.set(i + 1, termID);
            output.collect(outputKey, outputValue);
          }
        }

        // for (int i = 0; i < numberOfTopics; i++) {
        // a *zero* topic index and a *positive* topic index indicates the output is a term for
        // alpha updating
        // outputKey.set(0, i + 1);
        // outputValue.set((Gamma.digamma(tempGamma[i]) - digammaSumGamma));
        // output.collect(outputKey, outputValue);
        // }
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
    for (int i = 0; i < numberOfTopics; i++) {
      // a *zero* topic index and a *positive* topic index indicates the output is a term for
      // alpha updating
      outputKey.set(0, i + 1);
      outputValue.set(totalAlphaSufficientStatistics[i]);
      outputCollector.collect(outputKey, outputValue);
      totalAlphaSufficientStatistics[i] = 0;
    }

    if (!directEmit) {
      double[] phi = null;
      itr = totalPhi.keySet().iterator();
      while (itr.hasNext()) {
        int termID = itr.next();
        phi = totalPhi.get(termID);
        for (int i = 0; i < numberOfTopics; i++) {
          outputValue.set(phi[i]);

          // a *positive* topic index indicates the output is a phi values
          outputKey.set(i + 1, termID);
          outputCollector.collect(outputKey, outputValue);
        }
      }
      totalPhi.clear();
    }
    multipleOutputs.close();
  }

  /**
   * @param numberOfTopics number of topics defined by the current latent Dirichlet allocation
   *        model.
   * @param termCounts the term counts associated with the term
   * @param logBeta the beta vector
   * @param digammaGamma the gamma vector
   * @param logPhi the phi vector, take note that phi vector will be updated accordingly.
   * @param phiSum a vector recording the sum of all the phi's over all the terms, seeded from the
   *        caller program, take note that phiSum vector will be updated accordingly.
   * @param phiWeightSum a vector recording the weighted sum of all the phi's over all the terms,
   *        seeded from the caller program, take note that phiWeightSum vector will be updated
   *        accordingly.
   * @param emptyPhiTable a boolean value indicates whether the phiSum and phiWeightSum vector will
   *        be reset, they will be reset if True, and not. otherwise
   * @param updateLogGamma the updated gamma vector, may or may not seeded from the caller program,
   *        take note that updateGamma vector will be updated accordingly
   * @return
   */
  public static double updatePhi(int numberOfTopics, int termCounts, double[] logBeta,
      double[] digammaGamma, double[] logPhi, double[] updateLogGamma) {
    double convergePhi = 0;

    // initialize the normalize factor and the phi vector
    // phi is initialized in log scale
    logPhi[0] = (logBeta[0] + digammaGamma[0]);
    double normalizeFactor = logPhi[0];

    // compute the K-dimensional vector phi iteratively
    for (int i = 1; i < numberOfTopics; i++) {
      logPhi[i] = (logBeta[i] + digammaGamma[i]);
      normalizeFactor = LogMath.add(normalizeFactor, logPhi[i]);
    }

    for (int i = 0; i < numberOfTopics; i++) {
      // normalize the K-dimensional vector phi scale the
      // K-dimensional vector phi with the term count
      logPhi[i] -= normalizeFactor;
      convergePhi += termCounts * Math.exp(logPhi[i]) * (logBeta[i] - logPhi[i]);
      logPhi[i] += Math.log(termCounts);

      // update the K-dimensional vector gamma with phi
      updateLogGamma[i] = LogMath.add(updateLogGamma[i], logPhi[i]);
    }

    return convergePhi;
  }

  /**
   * Retrieve the beta array given the beta map and term index. If {@code beta} is null or
   * {@code termID} was not found in {@code beta}, this method will pop a message to
   * {@link System.out} and initialize it to avoid duplicate initialization in the future.
   * 
   * @param numberOfTopics number of topics defined by the current latent Dirichlet allocation
   *        model.
   * @param beta a {@link HMapIV<double[]>} object stores the beta matrix, the hash map is keyed by
   *        term index and valued by a corresponding double array
   * @param termID term index
   * @param numberOfTerms size of vocabulary in the whole corpus, used to initialize beta of the
   *        unloaded or non-initialized terms.
   * @return a double array of size {@link numberOfTopics} that stores the beta value of term index
   *         in log scale.
   */
  public static double[] retrieveBeta(int numberOfTopics, HMapIV<double[]> beta, int termID,
      int numberOfTerms) {
    Preconditions.checkArgument(beta != null, "Beta matrix was not properly initialized...");

    if (!beta.containsKey(termID)) {
      System.out.println("Term " + termID + " not found in the corresponding beta matrix...");

      double[] tempBeta = new double[numberOfTopics];
      for (int i = 0; i < numberOfTopics; i++) {
        // beta is initialized in log scale
        tempBeta[i] = Math.log(2 * Math.random() / numberOfTerms + Math.random());
        // tempBeta[i] = Math.log(1.0 / numberOfTerms + Math.random());
      }
      beta.put(termID, tempBeta);
    }

    return beta.get(termID);
  }

  /**
   * 
   * @param sequenceFileReader
   * @param numberOfTopics
   * @param numberOfTerms
   * @return
   * @throws IOException
   */
  // public static HMapIV<double[]> importBeta(SequenceFile.Reader sequenceFileReader,
  // int numberOfTopics, int numberOfTerms, boolean approximateBeta) throws IOException {
  public static HMapIV<double[]> importBeta(SequenceFile.Reader sequenceFileReader,
      int numberOfTopics, int numberOfTerms) throws IOException {
    HMapIV<double[]> beta = new HMapIV<double[]>();

    PairOfIntFloat pairOfIntFloat = new PairOfIntFloat();

    HMapIDW hashMap = new HMapIDW();
    // HashMap hashMap = new HashMap();

    // ProbDist hashMap = null;
    // if (!approximateBeta) {
    // hashMap = new HashMap();
    // } else {
    // hashMap = new BloomMap();
    // }

    while (sequenceFileReader.next(pairOfIntFloat, hashMap)) {
      Preconditions.checkArgument(
          pairOfIntFloat.getLeftElement() > 0 && pairOfIntFloat.getLeftElement() <= numberOfTopics,
          "Invalid beta vector for term " + pairOfIntFloat.getLeftElement() + "...");

      // topic is from 1 to K
      int topicIndex = pairOfIntFloat.getLeftElement() - 1;
      double logNormalizer = pairOfIntFloat.getRightElement();
      // double logNormalizer = Math.log(pairOfIntFloat.getRightElement());
      // double logNormalizer = Math.log(hashMap.getNormalizeFactor());

      // logNormalizer = LogMath.add(pairOfIntFloat.getRightElement(),
      // Settings.DEFAULT_LOG_ETA + Math.log(numberOfTerms));

      Iterator<Integer> itr = hashMap.keySet().iterator();
      while (itr.hasNext()) {
        int termIndex = itr.next();
        double logBetaValue = hashMap.get(termIndex);
        // double logBetaValue = Math.log(hashMap.get(termIndex));

        logBetaValue -= logNormalizer;

        if (!beta.containsKey(termIndex)) {
          double[] vector = new double[numberOfTopics];
          // this introduces some normalization error into the system, since beta might not be a
          // valid probability distribution anymore, normalizer may exclude some of those terms

          // for (int i = 0; i < vector.length; i++) {
          // vector[i] = Settings.DEFAULT_LOG_ETA;
          // }
          // vector[topicIndex] = LogMath.add(logBetaValue, vector[topicIndex]);

          vector[topicIndex] = logBetaValue;
          beta.put(termIndex, vector);
        } else {
          // beta.get(termIndex)[topicIndex] = betaValue;
          beta.get(termIndex)[topicIndex] = LogMath.add(logBetaValue,
              beta.get(termIndex)[topicIndex]);
        }
      }
    }

    return beta;
  }
}