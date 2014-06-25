package cc.mrlda;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.FileMerger;
import edu.umd.cloud9.io.map.HMapIDW;
import edu.umd.cloud9.io.pair.PairOfIntFloat;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.math.Gamma;

/**
 * This is the entry point of vanilla MapReduce latent Dirichlet allocation package.
 * 
 * @author kzhai
 */
public class VariationalInference extends Configured implements Tool {
  final Logger sLogger = Logger.getLogger(VariationalInference.class);

  static enum ParameterCounter {
    TOTAL_DOCS, TOTAL_TERMS, LOG_LIKELIHOOD, CONFIG_TIME, TRAINING_TIME, DUMMY_COUNTER,
  }

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    // Appender loggingAppender = new WriterAppender(new TTCCLayout(), new FileOutputStream(
    // VariationalInference.class.getSimpleName() + ".log." + variationalOptions.toString() + "."
    // + VariationalInferenceOptions.getDateTime(), true));
    // sLogger.addAppender(loggingAppender);

    return run(getConf(), new VariationalInferenceOptions(args));

    // return run(configuration, inputPath, outputPath, numberOfTopics, numberOfTerms,
    // numberOfIterations, mapperTasks, reducerTasks, localMerge, training, randomStartGamma,
    // resume, informedPrior, modelPath, snapshotIndex, directEmit, truncateBeta);
  }

  // private int run(Configuration configuration, VariationalOptions variationalOptions, String
  // inputPath, String outputPath,
  // int numberOfTopics, int numberOfTerms, int numberOfIterations, int mapperTasks,
  // int reducerTasks, boolean localMerge, boolean training, boolean randomStartGamma,
  // boolean resume, Path informedPrior, String modelPath, int snapshotIndex, boolean directEmit,
  // boolean truncateBeta) throws Exception {

  private int run(Configuration configuration, VariationalInferenceOptions variationalOptions)
      throws Exception {
    String inputPath = variationalOptions.getInputPath();
    String outputPath = variationalOptions.getOutputPath();
    int numberOfTopics = variationalOptions.getNumberOfTopics();
    int numberOfTerms = variationalOptions.getNumberOfTerms();
    int numberOfIterations = variationalOptions.getNumberOfIterations();
    int mapperTasks = variationalOptions.getMapperTasks();
    int reducerTasks = variationalOptions.getReducerTasks();
    boolean localMerge = variationalOptions.isLocalMerge();
    boolean training = variationalOptions.isTraining();
    boolean randomStartGamma = variationalOptions.isRandomStartGamma();
    boolean resume = variationalOptions.isResume();
    Path informedPrior = variationalOptions.getInformedPrior();
    String modelPath = variationalOptions.getModelPath();
    int snapshotIndex = variationalOptions.getSnapshotIndex();
    boolean directEmit = variationalOptions.isDirectEmit();
    boolean symmetricAlpha = variationalOptions.isSymmetricAlpha();

    boolean truncateBeta = variationalOptions.isTruncateBeta();

    sLogger.info("Tool: " + VariationalInference.class.getSimpleName());
    sLogger.info(" - input path: " + inputPath);
    sLogger.info(" - output path: " + outputPath);
    sLogger.info(" - number of topics: " + numberOfTopics);
    sLogger.info(" - number of terms: " + numberOfTerms);
    sLogger.info(" - number of iterations: " + numberOfIterations);
    sLogger.info(" - number of mappers: " + mapperTasks);
    sLogger.info(" - number of reducers: " + reducerTasks);
    sLogger.info(" - local merge: " + localMerge);
    sLogger.info(" - training mode: " + training);
    sLogger.info(" - random start gamma: " + randomStartGamma);
    sLogger.info(" - resume training: " + resume);
    sLogger.info(" - direct emit from mapper: " + directEmit);
    sLogger.info(" - truncation beta: " + truncateBeta);
    sLogger.info(" - informed prior: " + informedPrior);
    sLogger.info(" - symmetric alpha: " + symmetricAlpha);

    JobConf conf = new JobConf(configuration, VariationalInference.class);
    FileSystem fs = FileSystem.get(conf);

    // delete the overall output path
    Path outputDir = new Path(outputPath);
    if (!resume && fs.exists(outputDir)) {
      fs.delete(outputDir, true);
      fs.mkdirs(outputDir);
    }

    if (informedPrior != null) {
      Path eta = informedPrior;
      Preconditions.checkArgument(fs.exists(informedPrior) && fs.isFile(informedPrior),
          "Illegal informed prior file: must be an existing file...");
      informedPrior = new Path(outputPath + InformedPrior.ETA);
      FileUtil.copy(fs, eta, fs, informedPrior, false, conf);
    }

    Path inputDir = new Path(inputPath);
    Path tempDir = new Path(outputPath + Settings.TEMP + FileMerger.generateRandomString());

    // delete the output directory if it exists already
    fs.delete(tempDir, true);

    Path alphaDir = null;
    Path betaDir = null;
    Path gammaDir = null;

    Path documentGlobDir = new Path(tempDir.toString() + Path.SEPARATOR + Settings.GAMMA
        + Settings.UNDER_SCORE + Settings.GAMMA + Settings.DASH + Settings.STAR);

    SequenceFile.Reader sequenceFileReader = null;
    SequenceFile.Writer sequenceFileWriter = null;

    // these parameters are NOT used at all in the case of testing mode
    String betaPath = outputPath + Settings.BETA + Settings.DASH;
    String betaGlobDir = tempDir.toString() + Path.SEPARATOR + Settings.BETA + Settings.UNDER_SCORE
        + Settings.BETA + Settings.DASH + Settings.STAR;

    String alphaPath = outputPath + Settings.ALPHA + Settings.DASH;
    Path alphaSufficientStatisticsDir = new Path(tempDir.toString() + Path.SEPARATOR + "part-00000");
    double[] alphaVector = new double[numberOfTopics];

    if (!training) {
      alphaDir = new Path(modelPath + Settings.ALPHA + Settings.DASH + snapshotIndex);
      betaDir = new Path(modelPath + Settings.BETA + Settings.DASH + snapshotIndex);
    } else {
      if (!resume) {
        // initialize alpha vector randomly - if it doesn't already exist
        alphaDir = new Path(alphaPath + 0);
        for (int i = 0; i < alphaVector.length; i++) {
          // alphaVector[i] = Math.random();
          alphaVector[i] = 1e-3;
        }
        try {
          sequenceFileWriter = new SequenceFile.Writer(fs, conf, alphaDir, IntWritable.class,
              DoubleWritable.class);
          exportAlpha(sequenceFileWriter, alphaVector);
        } finally {
          IOUtils.closeStream(sequenceFileWriter);
        }
      } else {
        alphaDir = new Path(alphaPath + snapshotIndex);
        betaDir = new Path(betaPath + snapshotIndex);

        inputDir = new Path(outputPath + Settings.GAMMA + Settings.DASH + snapshotIndex);
      }
    }

    double lastLogLikelihood = 0;
    int iterationCount = snapshotIndex;
    int numberOfDocuments = 0;

    do {
      conf = new JobConf(configuration, VariationalInference.class);
      if (training) {
        conf.setJobName(VariationalInference.class.getSimpleName() + " - Iteration "
            + (iterationCount + 1));
      } else {
        conf.setJobName(VariationalInference.class.getSimpleName() + " - Test");
      }
      fs = FileSystem.get(conf);

      if (iterationCount != 0) {
        Preconditions.checkArgument(fs.exists(betaDir), "Missing model parameter beta...");
        DistributedCache.addCacheFile(betaDir.toUri(), conf);
      }
      Preconditions.checkArgument(fs.exists(alphaDir), "Missing model parameter alpha...");
      DistributedCache.addCacheFile(alphaDir.toUri(), conf);

      if (informedPrior != null) {
        Preconditions.checkArgument(fs.exists(informedPrior), "Informed prior does not exist...");
        DistributedCache.addCacheFile(informedPrior.toUri(), conf);
      }

      // conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.gamma",
      // Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_THRESHOLD);
      // conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.likelihood",
      // Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_CRITERIA);
      conf.setInt(Settings.PROPERTY_PREFIX + "model.mapper.converge.iteration",
          Settings.MAXIMUM_LOCAL_ITERATION);

      conf.setInt(Settings.PROPERTY_PREFIX + "model.topics", numberOfTopics);
      conf.setInt(Settings.PROPERTY_PREFIX + "corpus.terms", numberOfTerms);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.train", training);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.random.start", randomStartGamma);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.informed.prior", informedPrior != null);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.mapper.direct.emit", directEmit);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.truncate.beta", truncateBeta);

      conf.setNumMapTasks(mapperTasks);
      conf.setNumReduceTasks(reducerTasks);

      if (training) {
        MultipleOutputs.addMultiNamedOutput(conf, Settings.BETA, SequenceFileOutputFormat.class,
            PairOfIntFloat.class, HMapIDW.class);
        // MultipleOutputs.addMultiNamedOutput(conf, Settings.BETA, SequenceFileOutputFormat.class,
        // PairOfIntFloat.class, ProbDist.class);
        // MultipleOutputs.addMultiNamedOutput(conf, Settings.BETA, SequenceFileOutputFormat.class,
        // PairOfIntFloat.class, HashMap.class);
        // MultipleOutputs.addMultiNamedOutput(conf, Settings.BETA, SequenceFileOutputFormat.class,
        // PairOfIntFloat.class, BloomMap.class);
      }

      if (!randomStartGamma || !training) {
        MultipleOutputs.addMultiNamedOutput(conf, Settings.GAMMA, SequenceFileOutputFormat.class,
            IntWritable.class, Document.class);
      }

      conf.setMapperClass(DocumentMapper.class);
      conf.setReducerClass(TermReducer.class);
      conf.setCombinerClass(TermCombiner.class);
      conf.setPartitionerClass(TermPartitioner.class);

      conf.setMapOutputKeyClass(PairOfInts.class);
      conf.setMapOutputValueClass(DoubleWritable.class);
      conf.setOutputKeyClass(IntWritable.class);
      conf.setOutputValueClass(DoubleWritable.class);

      FileInputFormat.setInputPaths(conf, inputDir);
      FileOutputFormat.setOutputPath(conf, tempDir);

      // suppress the empty part files
      conf.setInputFormat(SequenceFileInputFormat.class);
      conf.setOutputFormat(SequenceFileOutputFormat.class);

      try {
        long startTime = System.currentTimeMillis();
        RunningJob job = JobClient.runJob(conf);
        sLogger.info("Iteration " + (iterationCount + 1) + " finished in "
            + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        Counters counters = job.getCounters();
        double logLikelihood = -counters.findCounter(ParameterCounter.LOG_LIKELIHOOD).getCounter()
            * 1.0 / Settings.DEFAULT_COUNTER_SCALE;
        sLogger.info("Log likelihood of the model is: " + logLikelihood);

        numberOfDocuments = (int) counters.findCounter(ParameterCounter.TOTAL_DOCS).getCounter();
        sLogger.info("Total number of documents is: " + numberOfDocuments);
        numberOfTerms = (int) (counters.findCounter(ParameterCounter.TOTAL_TERMS).getCounter() / numberOfTopics);
        sLogger.info("Total number of terms is: " + numberOfTerms);

        double configurationTime = counters.findCounter(ParameterCounter.CONFIG_TIME).getCounter()
            * 1.0 / numberOfDocuments;
        sLogger.info("Average time elapsed for mapper configuration (ms): " + configurationTime);
        double trainingTime = counters.findCounter(ParameterCounter.TRAINING_TIME).getCounter()
            * 1.0 / numberOfDocuments;
        sLogger.info("Average time elapsed for processing a document (ms): " + trainingTime);

        // break out of the loop if in testing mode
        if (training) {
          // update alpha only in training mode
          try {
            // load old alpha's into the system
            sequenceFileReader = new SequenceFile.Reader(fs, alphaDir, conf);
            alphaVector = importAlpha(sequenceFileReader, numberOfTopics);
            sLogger.info("Successfully import old alpha vector from file " + alphaDir);

            // load alpha sufficient statistics into the system
            double[] alphaSufficientStatistics = null;
            sequenceFileReader = new SequenceFile.Reader(fs, alphaSufficientStatisticsDir, conf);
            alphaSufficientStatistics = importAlpha(sequenceFileReader, numberOfTopics);
            sLogger.info("Successfully import alpha sufficient statistics tokens from file "
                + alphaSufficientStatisticsDir);

            // TODO: add option to stop alpha updating
            // update alpha
            if (symmetricAlpha) {
              double totalAlphaSufficientStatistics = 0;
              double oldAlpha = 0;
              for (int i = 0; i < numberOfTopics; i++) {
                totalAlphaSufficientStatistics += alphaSufficientStatistics[i];
                oldAlpha += alphaVector[i];
              }
              oldAlpha /= numberOfTopics;
              double newAlpha = updateScalarAlpha(numberOfTopics, numberOfDocuments, oldAlpha,
                  totalAlphaSufficientStatistics);
              for (int i = 0; i < numberOfTopics; i++) {
                alphaVector[i] = newAlpha;
              }
            } else {
              alphaVector = updateVectorAlpha(numberOfTopics, numberOfDocuments, alphaVector,
                  alphaSufficientStatistics);
            }
            sLogger.info("Successfully update new alpha vector.");

            // output the new alpha's to the system
            alphaDir = new Path(alphaPath + (iterationCount + 1));
            sequenceFileWriter = new SequenceFile.Writer(fs, conf, alphaDir, IntWritable.class,
                DoubleWritable.class);
            exportAlpha(sequenceFileWriter, alphaVector);
            sLogger.info("Successfully export new alpha vector to file " + alphaDir);
          } finally {
            // remove all the alpha sufficient statistics
            fs.deleteOnExit(alphaSufficientStatisticsDir);

            IOUtils.closeStream(sequenceFileReader);
            IOUtils.closeStream(sequenceFileWriter);
          }

          // merge beta's
          // TODO: local merge doesn't compress data
          if (localMerge) {
            throw new IOException("Please disable local merge option...");
            // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            // 0,
            // PairOfIntFloat.class, HMapIDW.class, true, true);

            // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            // 0,
            // PairOfIntFloat.class, ProbDist.class, true, true);
            // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            // 0,
            // PairOfIntFloat.class, BloomMap.class, true, true);
            // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            // 0,
            // PairOfIntFloat.class, HashMap.class, true, true);
          } else {
            betaDir = FileMerger.mergeSequenceFiles(new Configuration(), betaGlobDir, betaPath
                + (iterationCount + 1), reducerTasks, PairOfIntFloat.class, HMapIDW.class, true,
                true);
            // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            // reducerTasks, PairOfIntFloat.class, ProbDist.class, true, true);
            // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            // reducerTasks, PairOfIntFloat.class, BloomMap.class, true, true);
            // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            // reducerTasks, PairOfIntFloat.class, HashMap.class, true, true);
          }
        }
        
        // merge gamma (for alpha update) first and move document to the correct directory
        if (!randomStartGamma || !training) {
          gammaDir = inputDir;
          inputDir = new Path(outputPath + Settings.GAMMA + Settings.DASH + (iterationCount + 1));

          // TODO: technically, can rename the entire directory at this point of time, but found out
          // there are a lot of "part-*" files left-over, and set the outputformat to
          // NullOutputFormat does not resolve this problem, hence, need to rename them one-by-one.

          // fs.rename(tempDir, inputDir);
          fs.mkdirs(inputDir);
          FileStatus[] fileStatus = fs.globStatus(documentGlobDir);
          for (FileStatus file : fileStatus) {
            Path newPath = new Path(inputDir.toString() + Path.SEPARATOR + file.getPath().getName());
            fs.rename(file.getPath(), newPath);
          }

          if (iterationCount != 0) {
            // remove old gamma and document output
            fs.delete(gammaDir, true);
          }
        }

        sLogger.info("Log likelihood after iteration " + (iterationCount + 1) + " is "
            + logLikelihood);
        if (Math.abs((lastLogLikelihood - logLikelihood) / lastLogLikelihood) <= Settings.DEFAULT_GLOBAL_CONVERGE_CRITERIA) {
          sLogger.info("Model converged after " + (iterationCount + 1) + " iterations...");
          break;
        }
        lastLogLikelihood = logLikelihood;

        iterationCount++;
      } finally {
        // delete the output directory after job
        fs.delete(tempDir, true);
      }
    } while (iterationCount < numberOfIterations);

    return 0;
  }

  /**
   * This method updates the hyper-parameter alpha vector of the topic Dirichlet prior, which is an
   * asymmetric Dirichlet prior.
   * 
   * @param numberOfTopics the number of topics
   * @param numberOfDocuments the number of documents in this corpus
   * @param alphaVector the current alpha vector
   * @param alphaSufficientStatistics the alpha sufficient statistics collected from the corpus
   * @return
   */
  public static double[] updateVectorAlpha(int numberOfTopics, int numberOfDocuments,
      double[] alphaVector, double[] alphaSufficientStatistics) {
    double[] alphaVectorUpdate = new double[numberOfTopics];
    double[] alphaGradientVector = new double[numberOfTopics];
    double[] alphaHessianVector = new double[numberOfTopics];

    int alphaUpdateIterationCount = 0;

    // update the alpha vector until converge
    boolean keepGoing = true;
    try {
      int decay = 0;

      double alphaSum = 0;
      for (int j = 0; j < numberOfTopics; j++) {
        alphaSum += alphaVector[j];
      }

      while (keepGoing) {
        double sumG_H = 0;
        double sum1_H = 0;

        for (int i = 0; i < numberOfTopics; i++) {
          // compute alphaGradient
          alphaGradientVector[i] = numberOfDocuments
              * (Gamma.digamma(alphaSum) - Gamma.digamma(alphaVector[i]))
              + alphaSufficientStatistics[i];

          // compute alphaHessian
          alphaHessianVector[i] = -numberOfDocuments * Gamma.trigamma(alphaVector[i]);

          if (alphaGradientVector[i] == Double.POSITIVE_INFINITY
              || alphaGradientVector[i] == Double.NEGATIVE_INFINITY) {
            throw new ArithmeticException("Invalid ALPHA gradient matrix...");
          }

          sumG_H += alphaGradientVector[i] / alphaHessianVector[i];
          sum1_H += 1 / alphaHessianVector[i];
        }

        double z = numberOfDocuments * Gamma.trigamma(alphaSum);
        double c = sumG_H / (1 / z + sum1_H);

        while (true) {
          boolean singularHessian = false;

          for (int i = 0; i < numberOfTopics; i++) {
            double stepSize = Math.pow(Settings.DEFAULT_ALPHA_UPDATE_DECAY_FACTOR, decay)
                * (alphaGradientVector[i] - c) / alphaHessianVector[i];
            if (alphaVector[i] <= stepSize) {
              // the current hessian matrix is singular
              singularHessian = true;
              break;
            }
            alphaVectorUpdate[i] = alphaVector[i] - stepSize;
          }

          if (singularHessian) {
            // we need to further reduce the step size
            decay++;

            // recover the old alpha vector
            alphaVectorUpdate = alphaVector;
            if (decay > Settings.DEFAULT_ALPHA_UPDATE_MAXIMUM_DECAY) {
              break;
            }
          } else {
            // we have successfully update the alpha vector
            break;
          }
        }

        // compute the alpha sum and check for alpha converge
        alphaSum = 0;
        keepGoing = false;
        for (int j = 0; j < numberOfTopics; j++) {
          alphaSum += alphaVectorUpdate[j];
          if (Math.abs((alphaVectorUpdate[j] - alphaVector[j]) / alphaVector[j]) >= Settings.DEFAULT_ALPHA_UPDATE_CONVERGE_THRESHOLD) {
            keepGoing = true;
          }
        }

        if (alphaUpdateIterationCount >= Settings.DEFAULT_ALPHA_UPDATE_MAXIMUM_ITERATION) {
          keepGoing = false;
        }

        if (decay > Settings.DEFAULT_ALPHA_UPDATE_MAXIMUM_DECAY) {
          break;
        }

        alphaUpdateIterationCount++;
        alphaVector = alphaVectorUpdate;
      }
    } catch (IllegalArgumentException iae) {
      System.err.println(iae.getMessage());
      iae.printStackTrace();
    } catch (ArithmeticException ae) {
      System.err.println(ae.getMessage());
      ae.printStackTrace();
    }

    return alphaVector;
  }

  /**
   * This method imports alpha vector from a sequence file.
   * 
   * @param sequenceFileReader the reader for a sequence file
   * @param numberOfTopics number of topics
   * @return
   * @throws IOException
   */
  public static double[] importAlpha(SequenceFile.Reader sequenceFileReader, int numberOfTopics)
      throws IOException {
    double[] alpha = new double[numberOfTopics];
    int counts = 0;

    IntWritable intWritable = new IntWritable();
    DoubleWritable doubleWritable = new DoubleWritable();

    while (sequenceFileReader.next(intWritable, doubleWritable)) {
      Preconditions.checkArgument(intWritable.get() > 0 && intWritable.get() <= numberOfTopics,
          "Invalid alpha index: must be an integer in (0, " + numberOfTopics + "]...");

      // topic is from 1 to K
      alpha[intWritable.get() - 1] = doubleWritable.get();
      counts++;
    }
    Preconditions.checkArgument(counts == numberOfTopics, "Invalid alpha vector...");

    return alpha;
  }

  /**
   * This methods export alpha vector to a sequence file.
   * 
   * @param sequenceFileWriter the writer for sequence file
   * @param alpha the alpha vector get exported
   * @throws IOException
   */
  public static void exportAlpha(SequenceFile.Writer sequenceFileWriter, double[] alpha)
      throws IOException {
    IntWritable intWritable = new IntWritable();
    DoubleWritable doubleWritable = new DoubleWritable();
    for (int i = 0; i < alpha.length; i++) {
      doubleWritable.set(alpha[i]);
      intWritable.set(i + 1);
      sequenceFileWriter.append(intWritable, doubleWritable);
    }
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new VariationalInference(), args);
    System.exit(res);
  }

  /**
   * @deprecated
   * @param numberOfTopics
   * @param numberOfDocuments
   * @param alphaInit
   * @param alphaSufficientStatistics
   * @return
   */
  public static double updateScalarAlpha(int numberOfTopics, int numberOfDocuments,
      double alphaInit, double alphaSufficientStatistics) {
    int alphaUpdateIterationCount = 0;
    double alphaGradient = 0;
    double alphaHessian = 0;

    // update the alpha vector until converge
    boolean keepGoing = true;
    double alphaUpdate = alphaInit;
    try {
      double alphaSum = alphaUpdate * numberOfTopics;

      while (keepGoing) {
        alphaUpdateIterationCount++;

        if (Double.isNaN(alphaUpdate) || Double.isInfinite(alphaUpdate)) {
          alphaInit *= Settings.DEFAULT_ALPHA_UPDATE_SCALE_FACTOR;
          alphaUpdate = alphaInit;
        }

        alphaSum = alphaUpdate * numberOfTopics;

        // compute alphaGradient
        alphaGradient = numberOfDocuments
            * (numberOfTopics * Gamma.digamma(alphaSum) - numberOfTopics
                * Gamma.digamma(alphaUpdate)) + alphaSufficientStatistics;

        // compute alphaHessian
        alphaHessian = numberOfDocuments
            * (numberOfTopics * numberOfTopics * Gamma.trigamma(alphaSum) - numberOfTopics
                * Gamma.trigamma(alphaUpdate));

        alphaUpdate = Math.exp(Math.log(alphaUpdate) - alphaGradient
            / (alphaHessian * alphaUpdate + alphaGradient));

        if (Math.abs(alphaGradient) < Settings.DEFAULT_ALPHA_UPDATE_CONVERGE_THRESHOLD) {
          break;
        }

        if (alphaUpdateIterationCount > Settings.DEFAULT_ALPHA_UPDATE_MAXIMUM_ITERATION) {
          break;
        }
      }
    } catch (IllegalArgumentException iae) {
      System.err.println(iae.getMessage());
      iae.printStackTrace();
    } catch (ArithmeticException ae) {
      System.err.println(ae.getMessage());
      ae.printStackTrace();
    }

    return alphaUpdate;
  }
}