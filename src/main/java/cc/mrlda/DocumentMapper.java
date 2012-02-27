package cc.mrlda;

import java.io.IOException;
import java.util.Hashtable;
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

import cc.mrlda.Settings.ParameterCounter;
import cc.mrlda.util.Approximation;
import cc.mrlda.util.LogMath;
import cern.jet.stat.Gamma;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.array.ArrayListOfIntsWritable;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.HMapIV;

public class DocumentMapper extends MapReduceBase implements
    Mapper<IntWritable, Document, PairOfInts, DoubleWritable> {
  private long configurationTime = 0;
  private long trainingTime = 0;

  private static HMapIV<double[]> beta = null;
  private static double[] alpha = null;

  private static int numberOfTopics = Settings.DEFAULT_NUMBER_OF_TOPICS;
  private static int numberOfTerms = 0;

  private static double localConvergeForGamma = Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_THRESHOLD;
  private static double localConvergeForCriteria = Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_CRITERIA;
  private static int localConvergeForIteration = Settings.DEFAULT_GAMMA_UPDATE_MAXIMUM_ITERATION;

  private static boolean learning = Settings.LEARNING_MODE;
  private static boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

  // public static boolean informedPrior = false;
  private static double likelihoodAlpha = 0;
  private static double alphaSum = 0;

  private PairOfInts outputKey = new PairOfInts();
  private DoubleWritable outputValue = new DoubleWritable();

  private static MultipleOutputs multipleOutputs;
  private static OutputCollector<IntWritable, Document> outputDocument;

  private double[] tempBeta = null;
  private float[] tempGamma = null;
  private float[] gammaPointer = null;
  private float[] updateGamma = null;
  private Hashtable<Integer, double[]> phiTable = null;

  private Iterator<Integer> itr = null;

  public void configure(JobConf conf) {
    configurationTime = System.currentTimeMillis();

    numberOfTerms = conf.getInt(Settings.PROPERTY_PREFIX + "corpus.terms", Integer.MAX_VALUE);
    numberOfTopics = conf.getInt(Settings.PROPERTY_PREFIX + "model.topics",
        Settings.DEFAULT_NUMBER_OF_TOPICS);
    learning = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.train", Settings.LEARNING_MODE);

    updateGamma = new float[numberOfTopics];
    phiTable = new Hashtable<Integer, double[]>();

    multipleOutputs = new MultipleOutputs(conf);

    localConvergeForGamma = conf.getFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.gamma",
        Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_THRESHOLD);
    localConvergeForCriteria = conf.getFloat(Settings.PROPERTY_PREFIX
        + "model.mapper.converge.likelihood", Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_CRITERIA);
    localConvergeForIteration = conf.getInt(Settings.PROPERTY_PREFIX
        + "model.mapper.converge.iteration", Settings.DEFAULT_GAMMA_UPDATE_MAXIMUM_ITERATION);

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
    reporter.incrCounter(ParameterCounter.TOTAL_DOC, 1);
    trainingTime = System.currentTimeMillis();

    double likelihoodPhi = 0;
    double likelihoodGamma = 0;

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

      if (Math.random() < 0.1) {
        reporter.incrCounter(ParameterCounter.DUMMY_COUNTER, 1);
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
    if (!randomStartGamma || !learning) {
      outputDocument = multipleOutputs.getCollector(Settings.GAMMA, Settings.GAMMA, reporter);
      value.setGamma(tempGamma);
      outputDocument.collect(key, value);
    }

    // accumulate the log_likelihood of current document to job counter.
    reporter.incrCounter(ParameterCounter.LOG_LIKELIHOOD,
        (long) (-lastDocumentLikelihood * Settings.DEFAULT_COUNTER_SCALE));

    trainingTime = System.currentTimeMillis() - trainingTime;
    reporter.incrCounter(ParameterCounter.TRAINING_TIME, trainingTime);
  }

  public void close() throws IOException {
    multipleOutputs.close();
  }

  /**
   * @param numberOfTopics number of topics defined by the current latent Dirichlet allocation
   *        model.
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

  /**
   * Retrieve the beta array given the beta map and term index. If {@link beta} is null or
   * {@link termID} was not found in {@link beta}, this method will pop a message to
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