package cc.mrlda;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Calendar;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.FileMerger;

public class VariationalInferenceOptions {
  final Logger sLogger = Logger.getLogger(VariationalInferenceOptions.class);

  public static final Calendar calendar = Calendar.getInstance();
  public static final DateFormat dateFormat = new SimpleDateFormat("yyMMdd-HHmmss-SS");

  public static final String TRUNCATE_BETA_OPTION = "truncatebeta";

  private boolean directEmit = false;
  private boolean truncateBeta = false;

  private String inputPath = null;
  private String outputPath = null;

  private boolean localMerge = FileMerger.LOCAL_MERGE;
  private boolean randomStartGamma = Settings.RANDOM_START_GAMMA;

  private int numberOfTopics = 0;
  private int numberOfIterations = Settings.DEFAULT_GLOBAL_MAXIMUM_ITERATION;
  private int mapperTasks = Settings.DEFAULT_NUMBER_OF_MAPPERS;
  private int reducerTasks = Settings.DEFAULT_NUMBER_OF_REDUCERS;

  private int numberOfTerms = 0;

  private boolean resume = Settings.RESUME;
  private String modelPath = null;
  private int snapshotIndex = 0;
  private boolean training = Settings.LEARNING_MODE;

  private boolean symmetricAlpha = false;

  private Path informedPrior = null;

  public VariationalInferenceOptions(String args[]) {
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
        .create(Settings.MODEL_INDEX));

    options.addOption(Settings.RANDOM_START_GAMMA_OPTION, false,
        "start gamma from random point every iteration");

    options.addOption(Settings.SYMMETRIC_ALPHA, false, "symmetric topic Dirichlet prior");

    // options.addOption(FileMerger.LOCAL_MERGE_OPTION, false,
    // "merge output files and parameters locally, recommend for small scale cluster");
    options.addOption(Settings.DIRECT_EMIT, false,
        "disable in-mapper-combiner, enable this option if memory is limited");

    // "minimum memory threshold is " + Settings.MEMORY_THRESHOLD + " bytes and up to top " +
    // Settings.TOP_WORDS_FOR_CACHING + " frequent words"

    // options.addOption(Settings.TRUNCATE_BETA_OPTION, false,
    // "enable beta truncation of top 1000");

    options.addOption(OptionBuilder
        .withArgName(Settings.INTEGER_INDICATOR)
        .hasArg()
        .withDescription(
            "number of reducers (default - " + Settings.DEFAULT_NUMBER_OF_REDUCERS + ")")
        .create(TRUNCATE_BETA_OPTION));

    CommandLineParser parser = new GnuParser();
    HelpFormatter formatter = new HelpFormatter();
    try {
      CommandLine line = parser.parse(options, args);

      if (line.hasOption(Settings.HELP_OPTION)) {
        ToolRunner.printGenericCommandUsage(System.out);
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
          Preconditions.checkArgument(numberOfIterations > 0, "Illegal settings for "
              + Settings.ITERATION_OPTION + " option: must be strictly positive...");
        } else {
          sLogger.info("Warning: " + Settings.ITERATION_OPTION + " ignored in testing mode...");
        }
      }

      if (line.hasOption(Settings.MODEL_INDEX)) {
        snapshotIndex = Integer.parseInt(line.getOptionValue(Settings.MODEL_INDEX));
        if (!line.hasOption(Settings.INFERENCE_MODE_OPTION)) {
          resume = true;
          Preconditions.checkArgument(snapshotIndex < numberOfIterations, "Option "
              + Settings.ITERATION_OPTION + " and option " + Settings.MODEL_INDEX
              + " do not agree with each other: option " + Settings.ITERATION_OPTION
              + " must be strictly larger than option " + Settings.MODEL_INDEX + "...");
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

      if (line.hasOption(Settings.SYMMETRIC_ALPHA)) {
        Preconditions.checkArgument(training, "Option " + Settings.SYMMETRIC_ALPHA
            + " ignored due to testing mode...");
        Preconditions.checkArgument(!resume, "Option " + Settings.SYMMETRIC_ALPHA
            + " ignored due to resume...");
        symmetricAlpha = true;
      }

      if (line.hasOption(FileMerger.LOCAL_MERGE_OPTION)) {
        if (training) {
          // TODO: local merge does not handle compressed data.
          // localMerge = true;
        } else {
          sLogger.info("Warning: " + FileMerger.LOCAL_MERGE_OPTION + " ignored in testing mode...");
        }
      }

      if (line.hasOption(Settings.DIRECT_EMIT)) {
        directEmit = true;
      }

      if (line.hasOption(TRUNCATE_BETA_OPTION)) {
        if (training) {
          truncateBeta = true;
        } else {
          sLogger.info("Warning: " + TRUNCATE_BETA_OPTION + " ignored in testing mode...");
        }
      }

      if (line.hasOption(Settings.TOPIC_OPTION)) {
        numberOfTopics = Integer.parseInt(line.getOptionValue(Settings.TOPIC_OPTION));
      } else {
        throw new ParseException("Parsing failed due to " + Settings.TOPIC_OPTION
            + " not initialized...");
      }
      Preconditions.checkArgument(numberOfTopics > 0, "Illegal settings for "
          + Settings.TOPIC_OPTION + " option: must be strictly positive...");

      // TODO: need to relax this contrain in the future
      if (line.hasOption(Settings.TERM_OPTION)) {
        numberOfTerms = Integer.parseInt(line.getOptionValue(Settings.TERM_OPTION));
      }
      Preconditions.checkArgument(numberOfTerms > 0, "Illegal settings for " + Settings.TERM_OPTION
          + " option: must be strictly positive...");

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
      Preconditions.checkArgument(mapperTasks > 0, "Illegal settings for " + Settings.MAPPER_OPTION
          + " option: must be strictly positive...");

      if (line.hasOption(Settings.REDUCER_OPTION)) {
        if (training) {
          reducerTasks = Integer.parseInt(line.getOptionValue(Settings.REDUCER_OPTION));
          Preconditions.checkArgument(reducerTasks > 0, "Illegal settings for "
              + Settings.REDUCER_OPTION + " option: must be strictly positive...");
        } else {
          reducerTasks = 0;
          sLogger.info("Warning: " + Settings.REDUCER_OPTION + " ignored in test mode...");
        }
      }
    } catch (ParseException pe) {
      sLogger.error(pe.getMessage());
      ToolRunner.printGenericCommandUsage(System.err);
      formatter.printHelp(VariationalInference.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      sLogger.error(nfe.getMessage());
      nfe.printStackTrace(System.err);
      System.exit(0);
    } catch (IllegalArgumentException iae) {
      sLogger.error(iae.getMessage());
      iae.printStackTrace(System.err);
      System.exit(0);
    }
  }

  public VariationalInferenceOptions(String inputPath, String outputPath, int numberOfTopics,
      int numberOfTerms, int numberOfIterations, int mapperTasks, int reducerTasks,
      boolean localMerge, boolean training, boolean randomStartGamma, boolean resume,
      Path informedPrior, String modelPath, int snapshotIndex, boolean directEmit,
      boolean truncateBeta) {
    this.inputPath = inputPath;
    this.outputPath = outputPath;
    this.numberOfTopics = numberOfTopics;
    this.numberOfTerms = numberOfTerms;
    this.numberOfIterations = numberOfIterations;
    this.mapperTasks = mapperTasks;
    this.reducerTasks = reducerTasks;
    this.localMerge = localMerge;
    this.training = training;
    this.randomStartGamma = randomStartGamma;
    this.resume = resume;
    this.informedPrior = informedPrior;
    this.modelPath = modelPath;
    this.snapshotIndex = snapshotIndex;
    this.directEmit = directEmit;
    this.truncateBeta = truncateBeta;
  }

  public boolean isDirectEmit() {
    return directEmit;
  }

  public boolean isTruncateBeta() {
    return truncateBeta;
  }

  public String getInputPath() {
    return inputPath;
  }

  public String getOutputPath() {
    return outputPath;
  }

  public boolean isLocalMerge() {
    return localMerge;
  }

  public boolean isRandomStartGamma() {
    return randomStartGamma;
  }

  public int getNumberOfTopics() {
    return numberOfTopics;
  }

  public int getNumberOfIterations() {
    return numberOfIterations;
  }

  public int getMapperTasks() {
    return mapperTasks;
  }

  public int getReducerTasks() {
    return reducerTasks;
  }

  public int getNumberOfTerms() {
    return numberOfTerms;
  }

  public boolean isResume() {
    return resume;
  }

  public String getModelPath() {
    return modelPath;
  }

  public int getSnapshotIndex() {
    return snapshotIndex;
  }

  public boolean isTraining() {
    return training;
  }

  public boolean isSymmetricAlpha() {
    return symmetricAlpha;
  }

  public Path getInformedPrior() {
    return informedPrior;
  }

  public static String getDateTime() {
    return dateFormat.format(calendar.getTime());
  }

  public String toString() {
    return "K" + numberOfTopics + "";
  }
}