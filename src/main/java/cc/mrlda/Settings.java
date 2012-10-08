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
  public static final String INDEX_OPTION = "index";

  public static final String MAPPER_OPTION = "mapper";
  public static final String REDUCER_OPTION = "reducer";

  public static final int DEFAULT_NUMBER_OF_MAPPERS = 100;
  public static final int DEFAULT_NUMBER_OF_REDUCERS = 50;
  public static final String DEFAULT_QUEUE_NAME = "default";

  public static final char SPACE = ' ';
  public static final char UNDER_SCORE = '_';
  public static final char TAB = '\t';
  public static final char DASH = '-';
  public static final char DOT = '.';
  public static final char STAR = '*';

  public static final String TOPIC_OPTION = "topic";
  public static final String TERM_OPTION = "term";
  public static final String ITERATION_OPTION = "iteration";

  public static final double DEFAULT_COUNTER_SCALE = 1e6;

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

  public static final int MAXIMUM_LOCAL_ITERATION = 100;
  public static final int BURN_IN_SWEEP = 5;
  public static final double DEFAULT_GLOBAL_CONVERGE_CRITERIA = 0.000001;

  public static final double DEFAULT_LOG_ETA = Math.log(1e-100);

  public static final float DEFAULT_ALPHA_UPDATE_CONVERGE_THRESHOLD = 0.000001f;
  public static final int DEFAULT_ALPHA_UPDATE_MAXIMUM_ITERATION = 1000;
  public static final int DEFAULT_ALPHA_UPDATE_MAXIMUM_DECAY = 10;
  public static final float DEFAULT_ALPHA_UPDATE_DECAY_FACTOR = 0.8f;

  /**
   * @deprecated
   */
  public static final int DEFAULT_ALPHA_UPDATE_SCALE_FACTOR = 10;

  /**
   * 
   */
  public static final String DIRECT_EMIT = "directemit";
  public static final boolean DEFAULT_DIRECT_EMIT = false;

  public static final int MEMORY_THRESHOLD = 64 * 1024 * 1024;
  public static final int TOP_WORDS_FOR_CACHING = 10000;

  // public static final int DEFAULT_MAPRED_TASK_TIMEOUT = 1000 * 60 * 60;

  /**
   * sub-interface must override this property
   */
  static final String PROPERTY_PREFIX = Settings.class.getPackage().getName() + "" + DOT;
}