package cc.mrlda;

import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
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
import edu.umd.cloud9.io.map.HMapIFW;
import edu.umd.cloud9.io.pair.PairOfIntFloat;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.math.Gamma;

public class VariationalInference extends Configured implements Tool, Settings {

  public static final double DEFAULT_ETA = Math.log(1e-8);

  public static final float DEFAULT_ALPHA_UPDATE_CONVERGE_THRESHOLD = 0.000001f;
  public static final int DEFAULT_ALPHA_UPDATE_MAXIMUM_ITERATION = 1000;

  public static final int DEFAULT_ALPHA_UPDATE_MAXIMUM_DECAY = 10;
  public static final float DEFAULT_ALPHA_UPDATE_DECAY_FACTOR = 0.8f;

  /**
   * @deprecated
   */
  public static final int DEFAULT_ALPHA_UPDATE_SCALE_FACTOR = 10;

  /**
   * @deprecated
   */
  public static final float DEFAULT_ALPHA_UPDATE_INITIAL = 100f;

  // specific settings
  public static final String TRUNCATE_BETA_OPTION = "truncatebeta";
  public static final String MAPPER_COMBINER_OPTION = "mappercombiner";
  // set the minimum memory threshold, in bytes
  public static final int MEMORY_THRESHOLD = 64 * 1024 * 1024;

  static final Logger sLogger = Logger.getLogger(VariationalInference.class);

  static enum ParameterCounter {
    TOTAL_DOC, TOTAL_TERM, LOG_LIKELIHOOD, CONFIG_TIME, TRAINING_TIME, DUMMY_COUNTER,
  }

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    Options options = new Options();
    options.addOption(Settings.HELP_OPTION, false, "print the help message");

    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input file or directory").isRequired().create(Settings.INPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("output directory").isRequired().create(Settings.OUTPUT_OPTION));

    // TODO: relax the term constrain
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("number of terms").isRequired().create(Settings.TERM_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("number of topics").isRequired().create(Settings.TOPIC_OPTION));

    options.addOption(OptionBuilder
        .withArgName(Settings.INTEGER_INDICATOR)
        .hasArg()
        .withDescription(
            "number of iterations (default - " + Settings.DEFAULT_GLOBAL_MAXIMUM_ITERATION + ")")
        .create(Settings.ITERATION_OPTION));
    options
        .addOption(OptionBuilder
            .withArgName(Settings.INTEGER_INDICATOR)
            .hasArg()
            .withDescription(
                "number of mappers (default - " + Settings.DEFAULT_NUMBER_OF_MAPPERS + ")")
            .create(Settings.MAPPER_OPTION));
    options.addOption(OptionBuilder
        .withArgName(Settings.INTEGER_INDICATOR)
        .hasArg()
        .withDescription(
            "number of reducers (default - " + Settings.DEFAULT_NUMBER_OF_REDUCERS + ")")
        .create(Settings.REDUCER_OPTION));

    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArgs()
        .withDescription("run program in inference mode, i.e. test held-out likelihood")
        .create(Settings.INFERENCE_MODE_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArgs()
        .withDescription("seed informed prior").create(InformedPrior.INFORMED_PRIOR_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("the iteration/index of current model parameters")
        .create(Settings.RESUME_OPTION));

    options.addOption(FileMerger.LOCAL_MERGE_OPTION, false,
        "merge output files and parameters locally, recommend for small scale cluster");
    options.addOption(Settings.RANDOM_START_GAMMA_OPTION, false,
        "start gamma from random point every iteration");

    options.addOption(VariationalInference.MAPPER_COMBINER_OPTION, false,
        "enable in-mapper-combiner");

    options.addOption(OptionBuilder
        .withArgName(Settings.INTEGER_INDICATOR)
        .hasArg()
        .withDescription(
            "number of reducers (default - " + Settings.DEFAULT_NUMBER_OF_REDUCERS + ")")
        .create(VariationalInference.TRUNCATE_BETA_OPTION));

    // options.addOption(Settings.TRUNCATE_BETA_OPTION, false,
    // "enable beta truncation of top 1000");

    boolean mapperCombiner = false;
    boolean truncateBeta = false;

    String inputPath = null;
    String outputPath = null;

    boolean localMerge = FileMerger.LOCAL_MERGE;
    boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

    int numberOfTopics = 0;// Settings.DEFAULT_NUMBER_OF_TOPICS;
    int numberOfIterations = Settings.DEFAULT_GLOBAL_MAXIMUM_ITERATION;
    int mapperTasks = Settings.DEFAULT_NUMBER_OF_MAPPERS;
    int reducerTasks = Settings.DEFAULT_NUMBER_OF_REDUCERS;

    int numberOfTerms = 0;

    boolean resume = Settings.RESUME;
    String modelPath = null;
    int snapshotIndex = 0;
    boolean training = Settings.LEARNING_MODE;

    Path informedPrior = null;

    CommandLineParser parser = new GnuParser();
    HelpFormatter formatter = new HelpFormatter();
    try {
      CommandLine line = parser.parse(options, args);

      if (line.hasOption(Settings.HELP_OPTION)) {
        formatter.printHelp(VariationalInference.class.getName(), options);
        System.exit(0);
      }

      if (line.hasOption(Settings.INPUT_OPTION)) {
        inputPath = line.getOptionValue(Settings.INPUT_OPTION);
      }

      if (line.hasOption(Settings.OUTPUT_OPTION)) {
        outputPath = line.getOptionValue(Settings.OUTPUT_OPTION);

        if (!outputPath.endsWith(Path.SEPARATOR)) {
          outputPath += Path.SEPARATOR;
        }
      }

      if (line.hasOption(Settings.ITERATION_OPTION)) {
        if (training) {
          numberOfIterations = Integer.parseInt(line.getOptionValue(Settings.ITERATION_OPTION));
        } else {
          sLogger.info("Warning: " + Settings.ITERATION_OPTION + " ignored in testing mode...");
        }
      }

      if (line.hasOption(Settings.RESUME_OPTION)) {
        snapshotIndex = Integer.parseInt(line.getOptionValue(Settings.RESUME_OPTION));
        if (!line.hasOption(Settings.INFERENCE_MODE_OPTION)) {
          resume = true;
          Preconditions.checkArgument(snapshotIndex < numberOfIterations);
        }
      }

      if (line.hasOption(Settings.INFERENCE_MODE_OPTION)) {
        if (!line.hasOption(Settings.RESUME_OPTION)) {
          throw new ParseException("Model index missing: " + Settings.RESUME_OPTION
              + " was not initialized...");
        }

        modelPath = line.getOptionValue(Settings.INFERENCE_MODE_OPTION);
        if (!modelPath.endsWith(Path.SEPARATOR)) {
          modelPath += Path.SEPARATOR;
        }
        training = false;
        resume = false;
      }

      if (line.hasOption(FileMerger.LOCAL_MERGE_OPTION)) {
        if (training) {
          // TODO: local merge does not handle compressed data.
          // localMerge = true;
        } else {
          sLogger.info("Warning: " + FileMerger.LOCAL_MERGE_OPTION + " ignored in testing mode...");
        }
      }

      if (line.hasOption(VariationalInference.MAPPER_COMBINER_OPTION)) {
        if (training) {
          mapperCombiner = true;
        } else {
          sLogger.info("Warning: " + VariationalInference.MAPPER_COMBINER_OPTION
              + " ignored in testing mode...");
        }
      }

      if (line.hasOption(VariationalInference.TRUNCATE_BETA_OPTION)) {
        if (training) {
          truncateBeta = true;
        } else {
          sLogger.info("Warning: " + VariationalInference.TRUNCATE_BETA_OPTION
              + " ignored in testing mode...");
        }
      }

      if (line.hasOption(Settings.TOPIC_OPTION)) {
        numberOfTopics = Integer.parseInt(line.getOptionValue(Settings.TOPIC_OPTION));
      }

      // TODO: need to relax this contrain in the future
      if (line.hasOption(Settings.TERM_OPTION)) {
        numberOfTerms = Integer.parseInt(line.getOptionValue(Settings.TERM_OPTION));
        Preconditions.checkArgument(numberOfTerms > 0, "Illegal settings for "
            + Settings.TERM_OPTION + " option: " + numberOfTerms);
      }

      if (line.hasOption(Settings.RANDOM_START_GAMMA_OPTION)) {
        if (training) {
          randomStartGamma = true;
        } else {
          sLogger.info("Warning: " + Settings.RANDOM_START_GAMMA_OPTION
              + " ignored in testing mode...");
        }
      }

      if (line.hasOption(InformedPrior.INFORMED_PRIOR_OPTION)) {
        if (training) {
          informedPrior = new Path(line.getOptionValue(InformedPrior.INFORMED_PRIOR_OPTION));
        } else {
          sLogger.info("Warning: " + InformedPrior.INFORMED_PRIOR_OPTION
              + " ignored in test mode...");
        }
      }

      if (line.hasOption(Settings.MAPPER_OPTION)) {
        mapperTasks = Integer.parseInt(line.getOptionValue(Settings.MAPPER_OPTION));
      }

      if (line.hasOption(Settings.REDUCER_OPTION)) {
        if (training) {
          reducerTasks = Integer.parseInt(line.getOptionValue(Settings.REDUCER_OPTION));
        } else {
          reducerTasks = 0;
          sLogger.info("Warning: " + Settings.REDUCER_OPTION + " ignored in test mode...");
        }
      }
    } catch (ParseException pe) {
      System.err.println(pe.getMessage());
      formatter.printHelp(VariationalInference.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      System.err.println(nfe.getMessage());
      System.exit(0);
    } catch (IllegalArgumentException iae) {
      System.err.println(iae.getMessage());
      System.exit(0);
    }

    return run(inputPath, outputPath, numberOfTopics, numberOfTerms, numberOfIterations,
        mapperTasks, reducerTasks, localMerge, training, randomStartGamma, resume, informedPrior,
        modelPath, snapshotIndex, mapperCombiner, truncateBeta);
  }

  private int run(String inputPath, String outputPath, int numberOfTopics, int numberOfTerms,
      int numberOfIterations, int mapperTasks, int reducerTasks, boolean localMerge,
      boolean training, boolean randomStartGamma, boolean resume, Path informedPrior,
      String modelPath, int snapshotIndex, boolean mapperCombiner, boolean truncateBeta)
      throws Exception {

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
    sLogger.info(" - in-mapper-combiner: " + mapperCombiner);
    sLogger.info(" - truncation beta: " + truncateBeta);
    sLogger.info(" - informed prior: " + informedPrior);

    JobConf conf = new JobConf(getConf());
    conf.setJarByClass(getClass());
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
          "Illegal informed prior file...");
      informedPrior = new Path(outputPath + InformedPrior.ETA);
      FileUtil.copy(fs, eta, fs, informedPrior, false, conf);
    }

    Path inputDir = new Path(inputPath);
    Path tempDir = new Path(outputPath + Settings.TEMP);

    Path alphaDir = null;
    Path betaDir = null;

    Path documentGlobDir = new Path(tempDir.toString() + Path.SEPARATOR + Settings.GAMMA
        + Settings.STAR);

    // these parameters are NOT used at all in the case of testing mode
    Path alphaSufficientStatisticsDir = new Path(tempDir.toString() + Path.SEPARATOR + "part-00000");
    String betaGlobDir = tempDir.toString() + Path.SEPARATOR + Settings.BETA + Settings.STAR;

    SequenceFile.Reader sequenceFileReader = null;
    SequenceFile.Writer sequenceFileWriter = null;

    String betaPath = outputPath + Settings.BETA;
    String alphaPath = outputPath + Settings.ALPHA;
    double[] alphaVector = new double[numberOfTopics];

    if (!training) {
      alphaDir = new Path(modelPath + Settings.BETA + snapshotIndex);
      betaDir = new Path(modelPath + Settings.ALPHA + snapshotIndex);
    } else {
      if (!resume) {
        // initialize alpha vector randomly - if it doesn't already exist
        alphaDir = new Path(alphaPath + 0);
        for (int i = 0; i < alphaVector.length; i++) {
          alphaVector[i] = Math.random();
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

        inputDir = new Path(outputPath + Settings.GAMMA + snapshotIndex);
      }
    }

    double lastLogLikelihood = 0;
    int iterationCount = snapshotIndex;
    int numberOfDocuments = 0;

    do {
      conf = new JobConf(getConf());
      conf.setJarByClass(getClass());
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
      conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.iteration",
          Settings.MAXIMUM_GAMMA_ITERATION);

      conf.setInt(Settings.PROPERTY_PREFIX + "model.topics", numberOfTopics);
      conf.setInt(Settings.PROPERTY_PREFIX + "corpus.terms", numberOfTerms);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.train", training);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.random.start", randomStartGamma);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.informed.prior", informedPrior != null);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.mapper.combiner", mapperCombiner);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.truncate.beta", truncateBeta
          && iterationCount >= 5);

      // conf.setInt("mapred.task.timeout", VariationalInference.DEFAULT_MAPRED_TASK_TIMEOUT);
      // conf.set("mapred.child.java.opts", "-Xmx2048m");

      conf.setNumMapTasks(mapperTasks);
      conf.setNumReduceTasks(reducerTasks);

      if (training) {
        MultipleOutputs.addMultiNamedOutput(conf, Settings.BETA, SequenceFileOutputFormat.class,
            PairOfIntFloat.class, HMapIFW.class);
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

      conf.setCompressMapOutput(false);
      FileOutputFormat.setCompressOutput(conf, true);

      FileInputFormat.setInputPaths(conf, inputDir);
      FileOutputFormat.setOutputPath(conf, tempDir);

      // suppress the empty part files
      conf.setInputFormat(SequenceFileInputFormat.class);
      conf.setOutputFormat(SequenceFileOutputFormat.class);

      // delete the output directory if it exists already
      fs.delete(tempDir, true);

      long startTime = System.currentTimeMillis();
      RunningJob job = JobClient.runJob(conf);
      sLogger.info("Iteration " + (iterationCount + 1) + " finished in "
          + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

      Counters counters = job.getCounters();
      double logLikelihood = -counters.findCounter(ParameterCounter.LOG_LIKELIHOOD).getCounter()
          * 1.0 / Settings.DEFAULT_COUNTER_SCALE;
      sLogger.info("Log likelihood of the model is: " + logLikelihood);

      numberOfDocuments = (int) counters.findCounter(ParameterCounter.TOTAL_DOC).getCounter();
      sLogger.info("Total number of documents is: " + numberOfDocuments);
      numberOfTerms = (int) (counters.findCounter(ParameterCounter.TOTAL_TERM).getCounter() / numberOfTopics);
      sLogger.info("Total number of term is: " + numberOfTerms);

      double configurationTime = counters.findCounter(ParameterCounter.CONFIG_TIME).getCounter()
          * 1.0 / numberOfDocuments;
      sLogger.info("Average time elapsed for mapper configuration (ms): " + configurationTime);
      double trainingTime = counters.findCounter(ParameterCounter.TRAINING_TIME).getCounter() * 1.0
          / numberOfDocuments;
      sLogger.info("Average time elapsed for processing a document (ms): " + trainingTime);

      // break out of the loop if in testing mode
      if (!training) {
        break;
      }

      // merge gamma (for alpha update) first and move document to the correct directory
      if (!randomStartGamma) {
        // TODO: resume got error
        if (iterationCount != 0) {
          // remove old gamma and document output
          fs.delete(inputDir, true);
        }
        inputDir = new Path(outputPath + Settings.GAMMA + (iterationCount + 1));

        fs.mkdirs(inputDir);
        FileStatus[] fileStatus = fs.globStatus(documentGlobDir);
        for (FileStatus file : fileStatus) {
          Path newPath = new Path(inputDir.toString() + Path.SEPARATOR + file.getPath().getName());
          fs.rename(file.getPath(), newPath);
        }
      }

      // update alpha's
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

        // update alpha
        alphaVector = updateVectorAlpha(numberOfTopics, numberOfDocuments, alphaVector,
            alphaSufficientStatistics);
        sLogger.info("Successfully update new alpha vector.");

        // output the new alpha's to the system
        alphaDir = new Path(alphaPath + (iterationCount + 1));
        sequenceFileWriter = new SequenceFile.Writer(fs, conf, alphaDir, IntWritable.class,
            DoubleWritable.class);
        exportAlpha(sequenceFileWriter, alphaVector);
        sLogger.info("Successfully export new alpha vector to file " + alphaDir);

        // remove all the alpha sufficient statistics
        fs.deleteOnExit(alphaSufficientStatisticsDir);
      } finally {
        IOUtils.closeStream(sequenceFileReader);
        IOUtils.closeStream(sequenceFileWriter);
      }

      // merge beta's
      // TODO: local merge doesn't compress data
      if (localMerge) {
        betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1), 0,
            PairOfIntFloat.class, HMapIFW.class, true, true);
      } else {
        betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            reducerTasks, PairOfIntFloat.class, HMapIFW.class, true, true);
      }

      sLogger.info("Log likelihood after iteration " + (iterationCount + 1) + " is "
          + logLikelihood);
      if (Math.abs((lastLogLikelihood - logLikelihood) / lastLogLikelihood) <= Settings.DEFAULT_GLOBAL_CONVERGE_CRITERIA) {
        sLogger.info("Model converged after " + (iterationCount + 1) + " iterations...");
        break;
      }
      lastLogLikelihood = logLikelihood;

      iterationCount++;
    } while (iterationCount < numberOfIterations);

    return 0;
  }

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
            double stepSize = Math.pow(VariationalInference.DEFAULT_ALPHA_UPDATE_DECAY_FACTOR,
                decay) * (alphaGradientVector[i] - c) / alphaHessianVector[i];
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
            if (decay > VariationalInference.DEFAULT_ALPHA_UPDATE_MAXIMUM_DECAY) {
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
          if (Math.abs((alphaVectorUpdate[j] - alphaVector[j]) / alphaVector[j]) >= VariationalInference.DEFAULT_ALPHA_UPDATE_CONVERGE_THRESHOLD) {
            keepGoing = true;
          }
        }

        if (alphaUpdateIterationCount >= VariationalInference.DEFAULT_ALPHA_UPDATE_MAXIMUM_ITERATION) {
          keepGoing = false;
        }

        if (decay > VariationalInference.DEFAULT_ALPHA_UPDATE_MAXIMUM_DECAY) {
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
          alphaInit *= VariationalInference.DEFAULT_ALPHA_UPDATE_SCALE_FACTOR;
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

        if (Math.abs(alphaGradient) < VariationalInference.DEFAULT_ALPHA_UPDATE_CONVERGE_THRESHOLD) {
          break;
        }

        if (alphaUpdateIterationCount > VariationalInference.DEFAULT_ALPHA_UPDATE_MAXIMUM_ITERATION) {
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

  public static double[] importAlpha(SequenceFile.Reader sequenceFileReader, int numberOfTopics)
      throws IOException {
    double[] alpha = new double[numberOfTopics];
    int counts = 0;

    IntWritable intWritable = new IntWritable();
    DoubleWritable doubleWritable = new DoubleWritable();

    while (sequenceFileReader.next(intWritable, doubleWritable)) {
      Preconditions.checkArgument(intWritable.get() > 0 && intWritable.get() <= numberOfTopics,
          "Invalid alpha index: " + intWritable.get() + "...");

      // topic is from 1 to K
      alpha[intWritable.get() - 1] = doubleWritable.get();
      counts++;
    }
    Preconditions.checkArgument(counts == numberOfTopics, "Invalid alpha vector...");

    return alpha;
  }

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
    int res = ToolRunner.run(new VariationalInference(), args);
    System.exit(res);
  }
}