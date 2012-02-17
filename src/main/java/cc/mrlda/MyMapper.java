package cc.mrlda;

import java.io.IOException;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.StringTokenizer;

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

import cc.mrlda.Settings.ParamCounter;
import cc.mrlda.util.Approximation;
import cc.mrlda.util.LogMath;
import cern.jet.stat.Gamma;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.array.ArrayListOfIntsWritable;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.HMapIV;

public class MyMapper extends MapReduceBase implements
    Mapper<IntWritable, LDADocument, PairOfInts, DoubleWritable> {
  public static HMapIV<double[]> beta = null;
  public static double[] alpha = null;

  public static int numberOfTopics = Settings.DEFAULT_NUMBER_OF_TOPICS;
  public static int numberOfTerms = 0;

  public static double localConvergeForGamma = Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_THRESHOLD;
  public static double localConvergeForCriteria = Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_CRITERIA;
  public static int localConvergeForIteration = Settings.DEFAULT_GAMMA_UPDATE_MAXIMUM_ITERATION;

  public static boolean learning = Settings.LEARNING_MODE;
  public static boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

  // public static boolean informedPrior = false;
  public static double likelihoodAlpha = 0;
  public static double alphaSum = 0;

  public PairOfInts outputKey = new PairOfInts();
  public DoubleWritable outputValue = new DoubleWritable();

  private MultipleOutputs multipleOutputs;
  OutputCollector<IntWritable, LDADocument> outputDocument;

  double[] tempBeta = null;
  float[] tempGamma = null;
  float[] updateGamma = null;
  float[] gammaPointer = null;

  StringTokenizer stk = null;
  Iterator<Integer> itr = null;

  @SuppressWarnings("deprecation")
  public void map(IntWritable key, LDADocument value,
      OutputCollector<PairOfInts, DoubleWritable> output, Reporter reporter) throws IOException {

    double likelihoodPhi = 0;
    double likelihoodGamma = 0;

    reporter.incrCounter(ParamCounter.TOTAL_DOC, 1);

    if (alpha == null) {
      alpha = new double[numberOfTopics];
      double alphaLnGammaSum = 0;
      for (int i = 0; i < numberOfTopics; i++) {
        alpha[i] = Math.random();
        alphaSum += alpha[i];
        alphaLnGammaSum += Gamma.logGamma(alpha[i]);
      }
      likelihoodAlpha = Gamma.logGamma(alphaSum) - alphaLnGammaSum;
    }

    // initialize tempGamma for computing
    if (value.getGamma() != null && value.getNumberOfTopics() == numberOfTopics
        && !randomStartGamma) {
      // TODO: set up mechanisms to prevent starting from some irrelevant gamma value
      tempGamma = value.getGamma();
    } else {
      tempGamma = new float[numberOfTopics];
      for (int i = 0; i < numberOfTopics; i++) {
        tempGamma[i] = (float) (alpha[i] + 1.0f * value.getNumberOfWords() / numberOfTopics);
      }
    }

    double[] phi = new double[numberOfTopics];
    Hashtable<Integer, double[]> phiTable = new Hashtable<Integer, double[]>();

    double sumGamma = 0;
    boolean keepGoing = true;
    double lastDocumentLikelihood = 1.0;
    // be careful when adjust this initial value
    int gammaUpdateIterationCount = 1;

    while (keepGoing) {
      likelihoodPhi = 0;
      likelihoodGamma = 0;

      phiTable.clear();
      for (int i = 0; i < numberOfTopics; i++) {
        updateGamma[i] = (float) Math.log(alpha[i]);
      }

      HMapII content = value.getContent();
      // TODO: add in null check for content
      itr = content.keySet().iterator();

      while (itr.hasNext()) {
        int termID = itr.next();
        int termCounts = content.get(termID);
        // acquire the corresponding beta vector for this term
        tempBeta = retrieveBeta(numberOfTopics, beta, termID, numberOfTerms);

        phi = new double[numberOfTopics];
        likelihoodPhi += updatePhi(numberOfTopics, termCounts, tempBeta, tempGamma, phi,
            updateGamma);

        phiTable.put(termID, phi);
      }

      // compute the sum of gamma vector
      sumGamma = 0;
      for (int i = 0; i < numberOfTopics; i++) {
        updateGamma[i] = (float) Math.exp(updateGamma[i]);
        sumGamma += updateGamma[i];
      }

      // check for local converge on gamma
      keepGoing = false;
      for (int i = 0; i < numberOfTopics; i++) {
        // compute the converge criteria
        if (Math.abs((updateGamma[i] - tempGamma[i]) / tempGamma[i]) >= localConvergeForGamma) {
          keepGoing = true;
        }
      }

      likelihoodGamma -= Gamma.logGamma(sumGamma);

      // check for local converge on converge criteria
      double updateDocumentLikelihood = likelihoodAlpha + likelihoodGamma + likelihoodPhi;
      if (Math.abs((updateDocumentLikelihood - lastDocumentLikelihood) / lastDocumentLikelihood) <= localConvergeForCriteria) {
        keepGoing = false;
      }
      lastDocumentLikelihood = updateDocumentLikelihood;

      // check for local converge on iteration
      if (gammaUpdateIterationCount >= localConvergeForIteration) {
        keepGoing = false;
      }
      gammaUpdateIterationCount++;

      // swap tempGamma and updateGamma: update tempGamma for next iteration and reuse tempGamma
      gammaPointer = tempGamma;
      tempGamma = updateGamma;
      updateGamma = gammaPointer;
    }

    // compute the sum of gamma vector
    sumGamma = 0;
    for (int i = 0; i < numberOfTopics; i++) {
      sumGamma += tempGamma[i];
    }

    if (learning) {
      itr = phiTable.keySet().iterator();
      while (itr.hasNext()) {
        int termID = itr.next();
        phi = phiTable.get(termID);
        for (int i = 0; i < numberOfTopics; i++) {
          outputValue.set(phi[i]);

          // a *positive* topic index indicates the output is a phi values
          outputKey.set(i + 1, termID);
          output.collect(outputKey, outputValue);
        }
      }

      for (int i = 0; i < numberOfTopics; i++) {
        // a *zero* topic index and a *positive* topic index indicates the output is a term for
        // alpha updating
        outputKey.set(0, i + 1);
        outputValue.set((Approximation.digamma(tempGamma[i]) - Approximation.digamma(sumGamma)));
        output.collect(outputKey, outputValue);
      }
    }

    // output the embedded updated gamma together with document
    if (!randomStartGamma) {
      outputDocument = multipleOutputs.getCollector(Settings.DOCUMENT, Settings.DOCUMENT, reporter);
      value.setGamma(tempGamma);
      outputDocument.collect(key, value);
    }

    // accumulate the log_likelihood of current document to job counter.
    reporter.incrCounter(ParamCounter.LOG_LIKELIHOOD,
        (long) (-lastDocumentLikelihood * Settings.DEFAULT_COUNTER_SCALE));
  }

  /**
   * @param numberOfTopics the total number of topics
   * @param termCounts the term counts associated with the term
   * @param beta the beta vector
   * @param gamma the gamma vector
   * @param phi the phi vector, take note that phi vector will be updated accordingly.
   * @param phiSum a vector recording the sum of all the phi's over all the terms, seeded from the
   *        caller program, take note that phiSum vector will be updated accordingly.
   * @param phiWeightSum a vector recording the weighted sum of all the phi's over all the terms,
   *        seeded from the caller program, take note that phiWeightSum vector will be updated
   *        accordingly.
   * @param emptyPhiTable a boolean value indicates whether the phiSum and phiWeightSum vector will
   *        be reset, they will be reset if True, and not. otherwise
   * @param updateGamma the updated gamma vector, may or may not seeded from the caller program,
   *        take note that updateGamma vector will be updated accordingly
   * @return
   */
  public static double updatePhi(int numberOfTopics, int termCounts, double[] beta, float[] gamma,
      double[] phi, float[] updateGamma) {
    double convergePhi = 0;

    // initialize the normalize factor and the phi vector
    // phi is initialized in log scale
    phi[0] = (beta[0] + Approximation.digamma(gamma[0]));
    double normalizeFactor = phi[0];

    // compute the K-dimensional vector phi iteratively
    for (int i = 1; i < numberOfTopics; i++) {
      phi[i] = (beta[i] + Approximation.digamma(gamma[i]));
      normalizeFactor = LogMath.add(normalizeFactor, phi[i]);
    }

    for (int i = 0; i < numberOfTopics; i++) {
      // normalize the K-dimensional vector phi scale the
      // K-dimensional vector phi with the term count
      phi[i] -= normalizeFactor;
      convergePhi += termCounts * Math.exp(phi[i]) * (beta[i] - phi[i]);
      phi[i] += Math.log(termCounts);

      // update the K-dimensional vector gamma with phi
      updateGamma[i] = (float) LogMath.add(updateGamma[i], phi[i]);
    }

    return convergePhi;
  }

  public static double[] retrieveBeta(int numberOfTopics, HMapIV<double[]> beta, int termID,
      int numberOfTerms) {
    if (beta == null) {
      System.out.println("Beta matrix was not initialized...");
      beta = new HMapIV<double[]>();
    }

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

  public void configure(JobConf conf) {
    numberOfTerms = conf.getInt(Settings.PROPERTY_PREFIX + "corpus.terms", Integer.MAX_VALUE);
    numberOfTopics = conf.getInt(Settings.PROPERTY_PREFIX + "model.topics",
        Settings.DEFAULT_NUMBER_OF_TOPICS);
    learning = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.train", Settings.LEARNING_MODE);

    updateGamma = new float[numberOfTopics];
    multipleOutputs = new MultipleOutputs(conf);

    localConvergeForGamma = conf.getFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.gamma",
        Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_THRESHOLD);
    localConvergeForCriteria = conf.getFloat(Settings.PROPERTY_PREFIX
        + "model.mapper.converge.likelihood", Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_CRITERIA);
    localConvergeForIteration = conf.getInt(Settings.PROPERTY_PREFIX
        + "model.mapper.converge.iteration", Settings.DEFAULT_GAMMA_UPDATE_MAXIMUM_ITERATION);

    Path[] inputFiles;

    SequenceFile.Reader sequenceFileReader = null;
    try {
      inputFiles = DistributedCache.getLocalCacheFiles(conf);
      // TODO: check for the missing columns...
      if (inputFiles != null) {
        for (Path path : inputFiles) {
          try {
            sequenceFileReader = new SequenceFile.Reader(FileSystem.getLocal(conf), path, conf);

            if (path.getName().startsWith(Settings.BETA)) {
              // TODO: check whether seeded beta is valid, i.e., a true probability distribution
              Preconditions.checkArgument(beta == null, "Beta matrix was initialized already...");
              beta = VariationalInference.importBeta(sequenceFileReader, numberOfTopics,
                  numberOfTerms);
            } else if (path.getName().startsWith(Settings.ALPHA)) {
              Preconditions.checkArgument(alpha == null, "Alpha vector was initialized already...");

              // TODO: check the validity of alpha
              alpha = VariationalInference.importAlpha(sequenceFileReader, numberOfTopics);
              double sumLnGammaAlpha = 0;
              for (double value : alpha) {
                sumLnGammaAlpha += Gamma.logGamma(value);
                alphaSum += value;
              }
              likelihoodAlpha = Gamma.logGamma(alphaSum) - sumLnGammaAlpha;
            } else if (path.getName().startsWith(Settings.ETA)) {
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
  }

  public void close() throws IOException {
    multipleOutputs.close();
  }

  /**
   * @deprecated
   * @param sequenceFileReader
   * @param numberOfTopics
   * @return
   * @throws IOException
   */
  public static HMapIV<double[]> parseEta(SequenceFile.Reader sequenceFileReader, int numberOfTopics)
      throws IOException {
    HMapIV<double[]> beta = new HMapIV<double[]>();

    IntWritable intWritable = new IntWritable();
    ArrayListOfIntsWritable arrayListOfInts = new ArrayListOfIntsWritable();

    // TODO: normalize eta
    while (sequenceFileReader.next(intWritable, arrayListOfInts)) {
      Preconditions.checkArgument(intWritable.get() > 0 && intWritable.get() <= numberOfTopics,
          "Invalid eta prior for term " + intWritable.get() + "...");

      // topic is from 1 to K
      int topicIndex = intWritable.get() - 1;

      Iterator<Integer> itr = arrayListOfInts.iterator();
      while (itr.hasNext()) {
        int wordIndex = itr.next();
        if (!beta.containsKey(wordIndex)) {
          double[] vector = new double[numberOfTopics];
          for (int i = 0; i < vector.length; i++) {
            vector[i] = Settings.DEFAULT_UNINFORMED_LOG_ETA;
          }
        }
        beta.get(wordIndex)[topicIndex] = Settings.DEFAULT_INFORMED_LOG_ETA;
      }
    }

    return beta;
  }
}