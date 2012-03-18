package cc.mrlda;

public class Settings extends cc.common.util.Settings {
  static final String PROPERTY_PREFIX = Settings.class.getPackage() + "" + DOT;

  public static final int DEFAULT_MAPRED_TASK_TIMEOUT = 30 * 60 * 1000;

  public static final String TRUNCATE_BETA_OPTION = "truncatebeta";
  public static final String MAPPER_COMBINER_OPTION = "mappercombiner";
  // set the minimum memory threshold, in bytes
  public static final int MEMORY_THRESHOLD = 64 * 1024 * 1024;

  public static final String TOPIC_OPTION = "topic";
  public static final String TERM_OPTION = "term";
  public static final String ITERATION_OPTION = "iteration";

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

  public static final double DEFAULT_ETA = Math.log(1e-8);

  public static final int MAXIMUM_GAMMA_ITERATION = 10;

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

  public static final double DEFAULT_GLOBAL_CONVERGE_CRITERIA = 0.000001;
  public static final double DEFAULT_COUNTER_SCALE = 100;

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