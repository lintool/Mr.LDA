package cc.mrlda.polylda;

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
import edu.umd.cloud9.io.triple.TripleOfInts;

/**
 * @author kzhai
 */

public class VariationalInference extends Configured implements Tool {
  static final Logger sLogger = Logger.getLogger(VariationalInference.class);

  static enum ParameterCounter {
    TOTAL_DOC, LOG_LIKELIHOOD, CONFIG_TIME, TRAINING_TIME, DUMMY_COUNTER,
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
        .withDescription("number of languages").create(Settings.LANGUAGE_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("an estimate on average of terms across all languages")
        .create(Settings.TERM_OPTION));
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
    // TODO: add in informed prior for every language
    // options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArgs()
    // .withDescription("seed informed prior").create(InformedPrior.INFORMED_PRIOR_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("the iteration/index of current model parameters")
        .create(Settings.MODEL_INDEX));

    options.addOption(Settings.RANDOM_START_GAMMA_OPTION, false,
        "start gamma from random point every iteration");

    // options.addOption(FileMerger.LOCAL_MERGE_OPTION, false,
    // "merge output files and parameters locally, recommend for small scale cluster");
    // options.addOption(Settings.MAPPER_COMBINER_OPTION, false, "enable in-mapper-combiner");
    // options.addOption(Settings.TRUNCATE_BETA_OPTION, false,
    // "enable beta truncation of top 1000");

    String inputPath = null;
    String outputPath = null;

    boolean localMerge = FileMerger.LOCAL_MERGE;
    boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

    int numberOfTopics = 0;
    int numberOfLanguages = 0;
    int numberOfIterations = Settings.DEFAULT_GLOBAL_MAXIMUM_ITERATION;
    int mapperTasks = Settings.DEFAULT_NUMBER_OF_MAPPERS;
    int reducerTasks = Settings.DEFAULT_NUMBER_OF_REDUCERS;

    int[] numberOfTerms = null;

    boolean resume = Settings.RESUME;
    String modelPath = null;
    int snapshotIndex = 0;
    boolean training = Settings.LEARNING_MODE;

    // boolean mapperCombiner = false;

    Configuration configuration = this.getConf();
    CommandLineParser parser = new GnuParser();
    HelpFormatter formatter = new HelpFormatter();
    try {
      CommandLine line = parser.parse(options, args);

      if (line.hasOption(Settings.HELP_OPTION)) {
        formatter.printHelp(VariationalInference.class.getName(), options);
        ToolRunner.printGenericCommandUsage(System.out);
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

      if (line.hasOption(Settings.MODEL_INDEX)) {
        snapshotIndex = Integer.parseInt(line.getOptionValue(Settings.MODEL_INDEX));
        if (!line.hasOption(Settings.INFERENCE_MODE_OPTION)) {
          resume = true;
        }
      }

      if (line.hasOption(Settings.INFERENCE_MODE_OPTION)) {
        if (!line.hasOption(Settings.MODEL_INDEX)) {
          throw new ParseException("Model index missing: " + Settings.MODEL_INDEX
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
          localMerge = true;
        } else {
          sLogger.info("Warning: " + FileMerger.LOCAL_MERGE_OPTION + " ignored in testing mode...");
        }
      }

      // if (line.hasOption(Settings.MAPPER_COMBINER_OPTION)) {
      // if (training) {
      // mapperCombiner = true;
      // } else {
      // sLogger.info("Warning: " + Settings.MAPPER_COMBINER_OPTION
      // + " ignored in testing mode...");
      // }
      // }

      // if (line.hasOption(Settings.TRUNCATE_BETA_OPTION)) {
      // if (training) {
      // truncateBeta = true;
      // } else {
      // sLogger.info("Warning: " + Settings.TRUNCATE_BETA_OPTION + " ignored in testing mode...");
      // }
      // }

      if (line.hasOption(Settings.TOPIC_OPTION)) {
        numberOfTopics = Integer.parseInt(line.getOptionValue(Settings.TOPIC_OPTION));
      } else {
        throw new ParseException("Parsing failed due to " + Settings.TOPIC_OPTION
            + " not initialized...");
      }
      Preconditions.checkArgument(numberOfTopics > 0, "Illegal settings for "
          + Settings.TOPIC_OPTION + " option: must be strictly positive...");

      if (line.hasOption(Settings.LANGUAGE_OPTION)) {
        numberOfLanguages = Integer.parseInt(line.getOptionValue(Settings.LANGUAGE_OPTION));
      } else {
        throw new ParseException("Parsing failed due to " + Settings.LANGUAGE_OPTION
            + " not initialized...");
      }
      Preconditions.checkArgument(numberOfLanguages > 0, "Illegal settings for "
          + Settings.LANGUAGE_OPTION + " option: must be strictly positive...");

      if (line.hasOption(Settings.ITERATION_OPTION)) {
        if (training) {
          numberOfIterations = Integer.parseInt(line.getOptionValue(Settings.ITERATION_OPTION));
        } else {
          sLogger.info("Warning: " + Settings.ITERATION_OPTION + " ignored in testing mode...");
        }
      }

      // TODO: need to relax this contrain in the future
      // TODO: this is impossible in multiple languages
      int averageNumberOfTerms = 100000;
      if (line.hasOption(Settings.TERM_OPTION)) {
        averageNumberOfTerms = Integer.parseInt(line.getOptionValue(Settings.TERM_OPTION));
      } else {
        throw new ParseException("Parsing failed due to " + Settings.TERM_OPTION
            + " not initialized...");
      }

      numberOfTerms = new int[numberOfLanguages];
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        numberOfTerms[languageIndex] = averageNumberOfTerms;
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

      if (line.hasOption(Settings.RANDOM_START_GAMMA_OPTION)) {
        if (training) {
          randomStartGamma = true;
        } else {
          sLogger.info("Warning: " + Settings.RANDOM_START_GAMMA_OPTION
              + " ignored in testing mode...");
        }
      }

      // if (line.hasOption(InformedPrior.INFORMED_PRIOR_OPTION)) {
      // if (training) {
      // informedPrior = new Path(line.getOptionValue(InformedPrior.INFORMED_PRIOR_OPTION));
      // } else {
      // sLogger.info("Warning: " + InformedPrior.INFORMED_PRIOR_OPTION
      // + " ignored in test mode...");
      // }
      // }
    } catch (ParseException pe) {
      System.err.println(pe.getMessage());
      formatter.printHelp(VariationalInference.class.getName(), options);
      ToolRunner.printGenericCommandUsage(System.err);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      System.err.println(nfe.getMessage());
      System.exit(0);
    }

    // return run(inputPath, outputPath, numberOfTopics, numberOfLanguages, numberOfTerms,
    // numberOfIterations, mapperTasks, reducerTasks, localMerge, training, randomStartGamma,
    // resume, informedPrior, modelPath, snapshotIndex, mapperCombiner, truncateBeta);

    return run(configuration, inputPath, outputPath, numberOfTopics, numberOfLanguages, numberOfTerms,
        numberOfIterations, mapperTasks, reducerTasks, localMerge, training, randomStartGamma,
        resume, modelPath, snapshotIndex);
  }

  private int run(Configuration configuration, String inputPath, String outputPath, int numberOfTopics, int numberOfLanguages,
      int[] numberOfTerms, int numberOfIterations, int mapperTasks, int reducerTasks,
      boolean localMerge, boolean training, boolean randomStartGamma, boolean resume,
      String modelPath, int snapshotIndex) throws Exception {

    sLogger.info("Tool: " + VariationalInference.class.getSimpleName());

    sLogger.info(" - input path: " + inputPath);
    sLogger.info(" - output path: " + outputPath);
    sLogger.info(" - number of topics: " + numberOfTopics);
    sLogger.info(" - number of languages: " + numberOfLanguages);
    for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
      sLogger.info(" - number of terms in language " + (languageIndex + 1) + " is: "
          + numberOfTerms[languageIndex]);
    }
    sLogger.info(" - number of iterations: " + numberOfIterations);
    sLogger.info(" - number of mappers: " + mapperTasks);
    sLogger.info(" - number of reducers: " + reducerTasks);
    sLogger.info(" - local merge: " + localMerge);
    sLogger.info(" - training mode: " + training);
    sLogger.info(" - random start gamma: " + randomStartGamma);
    sLogger.info(" - resume training: " + resume);
    // sLogger.info(" - in-mapper-combiner: " + mapperCombiner);
    // sLogger.info(" - truncation beta: " + truncateBeta);
    // sLogger.info(" - informed prior: " + informedPrior);

    JobConf conf = new JobConf(configuration, VariationalInference.class);
    FileSystem fs = FileSystem.get(conf);

    // delete the overall output path
    Path outputDir = new Path(outputPath);
    if (!resume && fs.exists(outputDir)) {
      fs.delete(outputDir, true);
      fs.mkdirs(outputDir);
    }

    // if (informedPrior != null) {
    // Path eta = informedPrior;
    // Preconditions.checkArgument(fs.exists(informedPrior) && fs.isFile(informedPrior),
    // "Illegal informed prior file...");
    // informedPrior = new Path(outputPath + InformedPrior.ETA);
    // FileUtil.copy(fs, eta, fs, informedPrior, false, conf);
    // }

    Path inputDir = new Path(inputPath);
    Path tempDir = new Path(outputPath + Settings.TEMP + FileMerger.generateRandomString());

    // delete the output directory if it exists already
    fs.delete(tempDir, true);

    Path gammaDir = null;
    Path alphaDir = null;
    Path[] betaDir = new Path[numberOfLanguages];

    // for (int l = 0; l < numberOfLanguages; l++) {
    // betaDir[l] = null;
    // }

    Path documentGlobDir = new Path(tempDir.toString() + Path.SEPARATOR + Settings.GAMMA
        + Settings.UNDER_SCORE + Settings.GAMMA + Settings.DASH + Settings.STAR);

    // these parameters are NOT used at all in the case of testing mode
    Path alphaSufficientStatisticsDir = new Path(tempDir.toString() + Path.SEPARATOR + "part-00000");
    String[] betaGlobDir = new String[numberOfLanguages];
    for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
      betaGlobDir[languageIndex] = tempDir.toString() + Path.SEPARATOR + Settings.BETA
          + Settings.UNDER_SCORE + Settings.LANGUAGE_INDICATOR + (languageIndex + 1)
          + Settings.DASH + Settings.STAR;
    }

    SequenceFile.Reader sequenceFileReader = null;
    SequenceFile.Writer sequenceFileWriter = null;

    String[] betaPath = new String[numberOfLanguages];
    for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
      betaPath[languageIndex] = outputPath + Settings.BETA + Settings.UNDER_SCORE
          + Settings.LANGUAGE_INDICATOR + (languageIndex + 1) + Settings.DASH;
    }
    String alphaPath = outputPath + Settings.ALPHA + Settings.DASH;
    double[] alphaVector = new double[numberOfTopics];

    if (!training) {
      alphaDir = new Path(modelPath + Settings.ALPHA + Settings.DASH + snapshotIndex);
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        betaDir[languageIndex] = new Path(modelPath + Settings.BETA + Settings.UNDER_SCORE
            + Settings.LANGUAGE_INDICATOR + (languageIndex + 1) + Settings.DASH + snapshotIndex);
      }
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
          cc.mrlda.VariationalInference.exportAlpha(sequenceFileWriter, alphaVector);
        } finally {
          IOUtils.closeStream(sequenceFileWriter);
        }
      } else {
        alphaDir = new Path(alphaPath + snapshotIndex);
        for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
          betaDir[languageIndex] = new Path(betaPath[languageIndex] + snapshotIndex);
        }
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
        for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
          Preconditions.checkArgument(fs.exists(betaDir[languageIndex]),
              "Missing model parameter beta for language " + (languageIndex + 1) + "...");
          DistributedCache.addCacheFile(betaDir[languageIndex].toUri(), conf);
        }
      }
      Preconditions.checkArgument(fs.exists(alphaDir), "Missing model parameter alpha...");
      DistributedCache.addCacheFile(alphaDir.toUri(), conf);

      // TODO: add informed prior for this
      // if (informedPrior != null) {
      // Preconditions.checkArgument(fs.exists(informedPrior), "Informed prior does not exist...");
      // DistributedCache.addCacheFile(informedPrior.toUri(), conf);
      // }

      // conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.gamma",
      // Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_THRESHOLD);
      // conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.likelihood",
      // Settings.DEFAULT_GAMMA_UPDATE_CONVERGE_CRITERIA);
      conf.setInt(Settings.PROPERTY_PREFIX + "model.mapper.converge.iteration",
          Settings.MAXIMUM_LOCAL_ITERATION);

      conf.setInt(Settings.PROPERTY_PREFIX + "model.topics", numberOfTopics);
      conf.setInt(Settings.PROPERTY_PREFIX + "model.languages", numberOfLanguages);
      // language index starts from 1
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        conf.setInt(Settings.PROPERTY_PREFIX + "corpus.terms" + Settings.DOT + (languageIndex + 1),
            numberOfTerms[languageIndex]);
      }
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.train", training);
      conf.setBoolean(Settings.PROPERTY_PREFIX + "model.random.start", randomStartGamma);
      // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.informed.prior", informedPrior != null);
      // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.mapper.combiner", mapperCombiner);
      // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.truncate.beta", truncateBeta
      // && iterationCount >= 10);

      conf.setNumMapTasks(mapperTasks);
      conf.setNumReduceTasks(reducerTasks);

      if (training) {
        MultipleOutputs.addMultiNamedOutput(conf, Settings.BETA, SequenceFileOutputFormat.class,
            PairOfIntFloat.class, HMapIDW.class);
      }

      if (!randomStartGamma || !training) {
        MultipleOutputs.addMultiNamedOutput(conf, Settings.GAMMA, SequenceFileOutputFormat.class,
            IntWritable.class, Document.class);
      }

      conf.setMapperClass(DocumentMapper.class);
      conf.setReducerClass(TermReducer.class);
      conf.setCombinerClass(TermCombiner.class);
      conf.setPartitionerClass(TermPartitioner.class);

      conf.setMapOutputKeyClass(TripleOfInts.class);
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

        numberOfDocuments = (int) counters.findCounter(ParameterCounter.TOTAL_DOC).getCounter();
        sLogger.info("Total number of documents is: " + numberOfDocuments);

        for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
          numberOfTerms[languageIndex] = (int) (counters.findCounter(
              ParseCorpus.TOTAL_TERMS_IN_LANGUAGE, Settings.LANGUAGE_OPTION + (languageIndex + 1))
              .getCounter() / numberOfTopics);
          sLogger.info("Total number of terms in language " + (languageIndex + 1) + " is: "
              + numberOfTerms[languageIndex]);
        }

        double configurationTime = counters.findCounter(ParameterCounter.CONFIG_TIME).getCounter()
            * 1.0 / numberOfDocuments;
        sLogger.info("Average time elapsed for mapper configuration (ms): " + configurationTime);
        double trainingTime = counters.findCounter(ParameterCounter.TRAINING_TIME).getCounter()
            * 1.0 / numberOfDocuments;
        sLogger.info("Average time elapsed for processing a document (ms): " + trainingTime);

        // break out of the loop if in testing mode
        if (!training) {
          break;
        }

        // merge gamma (for alpha update) first and move document to the correct directory
        if (!randomStartGamma) {
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

        // update alpha's
        try {
          // load old alpha's into the system
          sequenceFileReader = new SequenceFile.Reader(fs, alphaDir, conf);
          alphaVector = cc.mrlda.VariationalInference.importAlpha(sequenceFileReader,
              numberOfTopics);
          sLogger.info("Successfully import old alpha vector from file " + alphaDir);

          // load alpha sufficient statistics into the system
          double[] alphaSufficientStatistics = null;
          sequenceFileReader = new SequenceFile.Reader(fs, alphaSufficientStatisticsDir, conf);
          alphaSufficientStatistics = cc.mrlda.VariationalInference.importAlpha(sequenceFileReader,
              numberOfTopics);
          sLogger.info("Successfully import alpha sufficient statistics tokens from file "
              + alphaSufficientStatisticsDir);

          // update alpha
          alphaVector = cc.mrlda.VariationalInference.updateVectorAlpha(numberOfTopics,
              numberOfDocuments, alphaVector, alphaSufficientStatistics);
          sLogger.info("Successfully update new alpha vector.");

          // output the new alpha's to the system
          alphaDir = new Path(alphaPath + (iterationCount + 1));
          sequenceFileWriter = new SequenceFile.Writer(fs, conf, alphaDir, IntWritable.class,
              DoubleWritable.class);
          cc.mrlda.VariationalInference.exportAlpha(sequenceFileWriter, alphaVector);
          sLogger.info("Successfully export new alpha vector to file " + alphaDir);

          // remove all the alpha sufficient statistics
          fs.deleteOnExit(alphaSufficientStatisticsDir);
        } finally {
          IOUtils.closeStream(sequenceFileReader);
          IOUtils.closeStream(sequenceFileWriter);
        }

        // merge beta's
        // TODO: local merge doesn't compress data
        // TODO: parallel this
        if (localMerge) {
          // TODO: disable this option if merge local
          // betaDir = FileMerger.mergeSequenceFiles(betaGlobDir, betaPath + (iterationCount + 1),
          // 0,
          // TripleOfInts.class, HMapIFW.class, true);
        } else {
          for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
            betaDir[languageIndex] = FileMerger.mergeSequenceFiles(new Configuration(),
                betaGlobDir[languageIndex], betaPath[languageIndex] + (iterationCount + 1),
                reducerTasks, PairOfIntFloat.class, HMapIDW.class, true, true);
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

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new VariationalInference(), args);
    System.exit(res);
  }
}