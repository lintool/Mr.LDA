package cc.mrlda;

public interface Settings {
  // common settings
  public static final String PATH_INDICATOR = "path";
  public static final String INTEGER_INDICATOR = "int";
  public static final String FLOAT_INDICATOR = "float";
  public static final String CLASS_INDICATOR = "class";

  public static final String HELP_OPTION = "help";

  public static final String INPUT_OPTION = "input";
  public static final String OUTPUT_OPTION = "output";

  public static final String MAPPER_OPTION = "mapper";
  public static final String REDUCER_OPTION = "reducer";

  public static final int DEFAULT_NUMBER_OF_MAPPERS = 100;
  public static final int DEFAULT_NUMBER_OF_REDUCERS = 50;

  public static final char SPACE = ' ';
  public static final char UNDER_SCORE = '_';
  public static final char TAB = '\t';
  public static final char DASH = '-';
  public static final char DOT = '.';
  public static final char STAR = '*';

  public static final String TOPIC_OPTION = "topic";
  public static final String TERM_OPTION = "term";
  public static final String ITERATION_OPTION = "iteration";

  public static final double DEFAULT_COUNTER_SCALE = 100;

  //

  public static final String INFERENCE_MODE_OPTION = "test";
  public static final String RANDOM_START_GAMMA_OPTION = "randomstart";
  public static final String RESUME_OPTION = "modelindex";

  // public static final int DEFAULT_NUMBER_OF_TOPICS = 100;
  public static final int DEFAULT_GLOBAL_MAXIMUM_ITERATION = 30;

  public static final boolean RANDOM_START_GAMMA = false;
  public static final boolean LEARNING_MODE = true;
  public static final boolean RESUME = false;

  public static final String TEMP = "temp";
  public static final String GAMMA = "gamma";
  public static final String BETA = "beta";
  public static final String ALPHA = "alpha";

  public static final int MAXIMUM_GAMMA_ITERATION = 100;
  public static final double DEFAULT_GLOBAL_CONVERGE_CRITERIA = 0.000001;

  /**
   * sub-interface must override this property
   */
  static final String PROPERTY_PREFIX = Settings.class.getPackage() + "" + DOT;

  // public static void exportSettings(SequenceFile.Writer sequenceFileWriter) {
  //
  // conf.setFloat(Settings.PROPERTY_PREFIX + "model.mapper.converge.iteration",
  // Settings.MAXIMUM_GAMMA_ITERATION);
  //
  // conf.setInt(Settings.PROPERTY_PREFIX + "model.topics", numberOfTopics);
  // conf.setInt(Settings.PROPERTY_PREFIX + "corpus.terms", numberOfTerms);
  // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.train", training);
  // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.random.start", randomStartGamma);
  // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.informed.prior", informedPrior != null);
  // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.mapper.combiner", mapperCombiner);
  // conf.setBoolean(Settings.PROPERTY_PREFIX + "model.truncate.beta", truncateBeta
  // && iterationCount >= 10);
  //
  // conf.setInt("mapred.task.timeout", Settings.DEFAULT_MAPRED_TASK_TIMEOUT);
  // conf.set("mapred.child.java.opts", "-Xmx2048m");
  //
  // }
}