package cc.mrlda;

import java.io.IOException;
import java.util.Iterator;

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

import cc.mrlda.util.Approximation;
import cc.mrlda.util.FileMerger;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapIFW;
import edu.umd.cloud9.io.map.HMapIIW;
import edu.umd.cloud9.io.pair.PairOfIntFloat;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.util.map.HMapIV;

public class VariationalInference extends Configured implements Tool {
  static final Logger sLogger = Logger.getLogger(VariationalInference.class);

  static enum ParameterCounter {
    TOTAL_DOC, TOTAL_TERM, LOG_LIKELIHOOD, CONFIG_TIME, TRAINING_TIME, DUMMY_COUNTER,
  }

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(Settings.HELP_OPTION, false, "print the help message");
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input file or directory").create(Settings.INPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("output directory").create(Settings.OUTPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("number of terms").create(Settings.TERM_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("number of topics (default - " + Settings.DEFAULT_NUMBER_OF_TOPICS + ")")
        .create(Settings.TOPIC_OPTION));
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

    String inputPath = null;
    String outputPath = null;

    boolean localMerge = FileMerger.LOCAL_MERGE;
    boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

    int numberOfTopics = Settings.DEFAULT_NUMBER_OF_TOPICS;
    int numberOfIterations = Settings.DEFAULT_GLOBAL_MAXIMUM_ITERATION;
    int mapperTasks = Settings.DEFAULT_NUMBER_OF_MAPPERS;
    int reducerTasks = Settings.DEFAULT_NUMBER_OF_REDUCERS;

    int numberOfTerms = 0;
    int numberOfDocuments = 0;

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
      } else {
        throw new ParseException("Parsing failed due to " + Settings.INPUT_OPTION
            + " not initialized...");
      }

      if (line.hasOption(Settings.OUTPUT_OPTION)) {
        outputPath = line.getOptionValue(Settings.OUTPUT_OPTION);

        if (!outputPath.endsWith(Path.SEPARATOR)) {
          outputPath += Path.SEPARATOR;
        }
      } else {
        throw new ParseException("Parsing failed due to " + Settings.OUTPUT_OPTION
            + " not initialized...");
      }

      if (line.hasOption(Settings.RESUME_OPTION)) {
        snapshotIndex = Integer.parseInt(line.getOptionValue(Settings.RESUME_OPTION));
        if (!line.hasOption(Settings.INFERENCE_MODE_OPTION)) {
          resume = true;
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
        localMerge = true;
      }

      if (line.hasOption(Settings.TOPIC_OPTION)) {
        numberOfTopics = Integer.parseInt(line.getOptionValue(Settings.TOPIC_OPTION));
      }

      if (line.hasOption(Settings.ITERATION_OPTION)) {
        numberOfIterations = Integer.parseInt(line.getOptionValue(Settings.ITERATION_OPTION));
      }

      // TODO: need to relax this contrain in the future
      if (line.hasOption(Settings.TERM_OPTION)) {
        numberOfTerms = Integer.parseInt(line.getOptionValue(Settings.TERM_OPTION));
      } else {
        throw new ParseException("Parsing failed due to " + Settings.TERM_OPTION
            + " not initialized...");
      }

      if (line.hasOption(Settings.MAPPER_OPTION)) {
        mapperTasks = Integer.parseInt(line.getOptionValue(Settings.MAPPER_OPTION));
      }

      if (line.hasOption(Settings.REDUCER_OPTION)) {
        if (training) {
          reducerTasks = Integer.parseInt(line.getOptionValue(Settings.REDUCER_OPTION));
        } else {
          sLogger.info("Warning: " + Settings.REDUCER_OPTION + " ignored in test mode...");
        }
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
        if (!training) {
          sLogger.info("Warning: " + InformedPrior.INFORMED_PRIOR_OPTION
              + " ignored in test mode...");
        } else {
          informedPrior = new Path(line.getOptionValue(InformedPrior.INFORMED_PRIOR_OPTION));
          // Preconditions.checkArgument(eta.getName().startsWith(InformedPrior.ETA));
        }
      }
    } catch (ParseException pe) {
      System.err.println(pe.getMessage());
      formatter.printHelp(VariationalInference.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      System.err.println(nfe.getMessage());
      System.exit(0);
    }

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
    sLogger.info(" - informed prior: " + informedPrior);

    JobConf conf = new JobConf(VariationalInference.class);
    FileSystem fs = FileSystem.get(conf);

    // delete the overall output path
    Path outputDir = new Path(outputPath);
    if (!resume && fs.exists(outputDir)) {
      fs.delete(outputDir, true);
      fs.mkdirs(outputDir);
    }

    Path eta = new Path(outputPath + InformedPrior.ETA);
    if (informedPrior != null) {
      Preconditions.checkArgument(fs.exists(informedPrior) && fs.isFile(informedPrior),
          "Illegal informed prior file...");
      FileUtil.copy(fs, informedPrior, fs, eta, false, conf);
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
      }
    }

    double lastLogLikelihood = 0;
    int iterationCount = snapshotIndex;

    do {
      conf = new JobConf(VariationalInference.class);
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

      if (eta != null) {
        Preconditions.checkArgument(fs.exists(eta), "Informed prior does not exist...");
        DistributedCache.addCacheFile(eta.toUri(), conf);
      }

      conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.gamma",
          Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_THRESHOLD);
      conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.likelihood",
          Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_CRITERIA);
      conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.iteration",
          Settings.DEFAULT_GAMMA_UPDATE_MAXIMUM_ITERATION);

      conf.setInt(Settings.PROPERTY_PREFIX + "model.topics", numberOfTopics);
      conf.setInt(Settings.PROPERTY_PREFIX + "corpus.terms", numberOfTerms);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.train", training);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.informed.prior", eta != null);

      conf.setNumMapTasks(mapperTasks);
      if (training) {
        conf.setNumReduceTasks(reducerTasks);
      } else {
        conf.setNumReduceTasks(0);
      }

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

      conf.setCompressMapOutput(true);
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
      sLogger.info("Iteration " + iterationCount + " finished in "
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
      if (localMerge) {
        betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1), 0,
            PairOfIntFloat.class, HMapIFW.class, true);
      } else {
        betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
            reducerTasks, PairOfIntFloat.class, HMapIIW.class, true);
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
              * (Approximation.digamma(alphaSum) - Approximation.digamma(alphaVector[i]))
              + alphaSufficientStatistics[i];

          // compute alphaHessian
          alphaHessianVector[i] = -numberOfDocuments * Approximation.trigamma(alphaVector[i]);

          if (alphaGradientVector[i] == Double.POSITIVE_INFINITY
              || alphaGradientVector[i] == Double.NEGATIVE_INFINITY) {
            throw new ArithmeticException("Invalid ALPHA gradient matrix...");
          }

          sumG_H += alphaGradientVector[i] / alphaHessianVector[i];
          sum1_H += 1 / alphaHessianVector[i];
        }

        double z = numberOfDocuments * Approximation.trigamma(alphaSum);
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
            * (numberOfTopics * Approximation.digamma(alphaSum) - numberOfTopics
                * Approximation.digamma(alphaUpdate)) + alphaSufficientStatistics;

        // compute alphaHessian
        alphaHessian = numberOfDocuments
            * (numberOfTopics * numberOfTopics * Approximation.trigamma(alphaSum) - numberOfTopics
                * Approximation.trigamma(alphaUpdate));

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

  public static HMapIV<double[]> importBeta(SequenceFile.Reader sequenceFileReader,
      int numberOfTopics, int numberOfTerms) throws IOException {
    HMapIV<double[]> beta = new HMapIV<double[]>();

    PairOfIntFloat pairOfIntFloat = new PairOfIntFloat();
    HMapIFW hMapIFW = new HMapIFW();

    while (sequenceFileReader.next(pairOfIntFloat, hMapIFW)) {
      Preconditions.checkArgument(
          pairOfIntFloat.getLeftElement() > 0 && pairOfIntFloat.getLeftElement() <= numberOfTopics,
          "Invalid beta vector for term " + pairOfIntFloat.getKey() + "...");

      // topic is from 1 to K
      int topicIndex = pairOfIntFloat.getLeftElement() - 1;
      double normalizer = pairOfIntFloat.getRightElement();

      Iterator<Integer> itr = hMapIFW.keySet().iterator();
      while (itr.hasNext()) {
        int termIndex = itr.next();
        double betaValue = hMapIFW.get(termIndex);

        betaValue -= normalizer;

        if (!beta.containsKey(termIndex)) {
          double[] vector = new double[numberOfTopics];
          // this introduces some normalization error into the system, since beta might not be a
          // valid probability distribution anymore, normalizer may exclude some of those terms
          for (int i = 0; i < vector.length; i++) {
            vector[i] = Math.log(1.0 / numberOfTerms);
          }
          vector[topicIndex] = betaValue;
          beta.put(termIndex, vector);
        } else {
          beta.get(termIndex)[topicIndex] = betaValue;
        }
      }
    }

    return beta;
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new VariationalInference(), args);
    System.exit(res);
  }
}