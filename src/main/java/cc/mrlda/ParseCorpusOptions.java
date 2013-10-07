package cc.mrlda;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.log4j.Logger;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.cn.smart.SmartChineseAnalyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.util.Version;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.FileMerger;

public class ParseCorpusOptions {
  final Logger sLogger = Logger.getLogger(ParseCorpusOptions.class);

  public static final String ANALYZER = "analyzer";
  public static final String STOP_LIST = "stoplist";
  public static final String INDEX = "index";

  public static final String MINIMUM_DOCUMENT_FREQUENCY = "minimumdocumentfrequency";
  public static final String MAXIMUM_DOCUMENT_FREQUENCY = "maximumdocumentfrequency";
  public static final String MINIMUM_TERM_FREQUENCY = "minimumtermfrequency";
  public static final String MAXIMUM_TERM_FREQUENCY = "maximumtermfrequency";

  public static final float DEFAULT_MINIMUM_DOCUMENT_FREQUENCY = 0.0f;
  public static final float DEFAULT_MAXIMUM_DOCUMENT_FREQUENCY = 1.0f;
  public static final float DEFAULT_MINIMUM_TERM_FREQUENCY = 0.0f;
  public static final float DEFAULT_MAXIMUM_TERM_FREQUENCY = 1.0f;

  private String inputPath = null;
  private String outputPath = null;
  private String indexPath = null;
  private Class<? extends Analyzer> analyzerClass = null;
  private int numberOfMappers = Settings.DEFAULT_NUMBER_OF_MAPPERS;
  private int numberOfReducers = Settings.DEFAULT_NUMBER_OF_REDUCERS;
  private float maximumDocumentFrequency = DEFAULT_MAXIMUM_DOCUMENT_FREQUENCY;
  private float minimumDocumentFrequency = DEFAULT_MINIMUM_DOCUMENT_FREQUENCY;
  // private int maximumDocumentFrequency = Integer.MAX_VALUE;
  // private int minimumDocumentFrequency = 0;
  private boolean localMerge = FileMerger.LOCAL_MERGE;
  private String stopListPath = null;

  public ParseCorpusOptions(String args[]) {
    Options options = new Options();

    options.addOption(Settings.HELP_OPTION, false, "print the help message");
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input file(s) or directory").isRequired().create(Settings.INPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("output directory").isRequired().create(Settings.OUTPUT_OPTION));

    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("existing term indices").create(INDEX));

    options.addOption(OptionBuilder
        .withArgName(Settings.CLASS_INDICATOR)
        .hasArg()
        .withDescription(
            "analyzer class in Lucene (e.g., " + StandardAnalyzer.class + ", or "
                + SmartChineseAnalyzer.class
                + "), default to be null, which tokenize a document by space").create(ANALYZER));

    options
        .addOption(FileMerger.LOCAL_MERGE_OPTION, false,
            "merge files locally, recommend for small scale cluster with a limited number of documents");

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

    options.addOption(OptionBuilder
        .withArgName(Settings.FLOAT_INDICATOR)
        .hasArg()
        .withDescription(
            "minimum document frequency (default - " + DEFAULT_MINIMUM_DOCUMENT_FREQUENCY + ")")
        .create(MINIMUM_DOCUMENT_FREQUENCY));
    options.addOption(OptionBuilder
        .withArgName(Settings.FLOAT_INDICATOR)
        .hasArg()
        .withDescription(
            "maximum document frequency (default - " + DEFAULT_MAXIMUM_DOCUMENT_FREQUENCY + ")")
        .create(MAXIMUM_DOCUMENT_FREQUENCY));

    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("stopword list").create(STOP_LIST));

    // options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
    // .withDescription("minimum document frequency (default - " + 0 + ")")
    // .create(MINIMUM_DOCUMENT_FREQUENCY));
    // options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
    // .withDescription("maximum document frequency (default - " + Integer.MAX_VALUE + ")")
    // .create(MAXIMUM_DOCUMENT_FREQUENCY));

    CommandLineParser parser = new GnuParser();
    HelpFormatter formatter = new HelpFormatter();
    try {
      CommandLine line = parser.parse(options, args);

      if (line.hasOption(Settings.HELP_OPTION)) {
        formatter.printHelp(ParseCorpus.class.getName(), options);
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
      } else {
        throw new ParseException("Parsing failed due to " + Settings.OUTPUT_OPTION
            + " not initialized...");
      }

      if (line.hasOption(STOP_LIST)) {
        stopListPath = line.getOptionValue(STOP_LIST);
      }

      if (line.hasOption(INDEX)) {
        indexPath = line.getOptionValue(INDEX);
      }

      if (line.hasOption(ANALYZER)) {
        analyzerClass = (Class<? extends Analyzer>) Class.forName(line.getOptionValue(ANALYZER));
        // Constructor cons = analyzerClass.getDeclaredConstructor(new Class[] { Version.class });
        Constructor cons = analyzerClass.getDeclaredConstructor(Version.class);
        Analyzer tempAnalyzer = (Analyzer) cons.newInstance(Version.LUCENE_40);

        // String[] examplesChinese = { "大家 晚上 好 ， 我 的 名字 叫 Ke Zhai 。",
        // "日本 人民 要 牢牢 记住 ： “ 钓鱼岛 是 中国 神圣 不可 分割 的 领土 。 ” （ 续 ）",
        // "中国 进出口 银行 最近 在 日本 取得 债券 信用 等级 aa - 。" };
        // for (String text : examplesChinese) {
        // sLogger.info("Analyzing \"" + text + "\"");
        // String name = tempAnalyzer.getClass().getSimpleName();
        // sLogger.info("\t" + name + ":");
        // sLogger.info("\t");
        // TokenStream stream = tempAnalyzer.tokenStream("contents,", new StringReader(text));
        // stream.reset();
        // CharTermAttribute charTermAttribute = stream.addAttribute(CharTermAttribute.class);
        // while (stream.incrementToken()) {
        // sLogger.info("[" + charTermAttribute.toString() + "] ");
        // }
        // sLogger.info("\n");
        // }
      }

      if (line.hasOption(FileMerger.LOCAL_MERGE_OPTION)) {
        localMerge = true;
      }

      if (line.hasOption(Settings.MAPPER_OPTION)) {
        numberOfMappers = Integer.parseInt(line.getOptionValue(Settings.MAPPER_OPTION));
      }

      if (line.hasOption(Settings.REDUCER_OPTION)) {
        numberOfReducers = Integer.parseInt(line.getOptionValue(Settings.REDUCER_OPTION));
      }

      if (line.hasOption(MINIMUM_DOCUMENT_FREQUENCY)) {
        minimumDocumentFrequency = Float
            .parseFloat(line.getOptionValue(MINIMUM_DOCUMENT_FREQUENCY));
        Preconditions.checkArgument(minimumDocumentFrequency >= 0 && minimumDocumentFrequency <= 1,
            "Illegal settings for " + MINIMUM_DOCUMENT_FREQUENCY + " option: must be in [0, 1]...");
      }

      if (line.hasOption(MAXIMUM_DOCUMENT_FREQUENCY)) {
        maximumDocumentFrequency = Float
            .parseFloat(line.getOptionValue(MAXIMUM_DOCUMENT_FREQUENCY));
        Preconditions.checkArgument(maximumDocumentFrequency >= 0 && maximumDocumentFrequency <= 1,
            "Illegal settings for " + MAXIMUM_DOCUMENT_FREQUENCY + " option: must be in [0, 1]...");
      }

      Preconditions.checkArgument(minimumDocumentFrequency < maximumDocumentFrequency, "Option "
          + MAXIMUM_DOCUMENT_FREQUENCY + " and option " + MINIMUM_DOCUMENT_FREQUENCY
          + " do not agree with each other: option " + MAXIMUM_DOCUMENT_FREQUENCY
          + " must be strictly larger than option " + MINIMUM_DOCUMENT_FREQUENCY + "...");
    } catch (ParseException pe) {
      sLogger.error(pe.getMessage());
      formatter.printHelp(ParseCorpus.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      sLogger.error(nfe.getMessage());
      nfe.printStackTrace(System.err);
      System.exit(0);
    } catch (ClassNotFoundException cnfe) {
      sLogger.error(cnfe.getMessage());
      cnfe.printStackTrace(System.err);
      System.exit(0);
    } catch (SecurityException se) {
      sLogger.error(se.getMessage());
      se.printStackTrace(System.err);
      System.exit(0);
    } catch (NoSuchMethodException nsme) {
      sLogger.error(nsme.getMessage());
      nsme.printStackTrace(System.err);
      System.exit(0);
    } catch (IllegalArgumentException iae) {
      sLogger.error(iae.getMessage());
      iae.printStackTrace(System.err);
      System.exit(0);
    } catch (InstantiationException ie) {
      sLogger.error(ie.getMessage());
      ie.printStackTrace(System.err);
      System.exit(0);
    } catch (IllegalAccessException iae) {
      sLogger.error(iae.getMessage());
      iae.printStackTrace(System.err);
      System.exit(0);
    } catch (InvocationTargetException ite) {
      sLogger.error(ite.getMessage());
      ite.printStackTrace(System.err);
      System.exit(0);
    }
  }

  public String getInputPath() {
    return inputPath;
  }

  public String getOutputPath() {
    return outputPath;
  }

  public String getStopListPath() {
    return stopListPath;
  }

  public String getIndexPath() {
    return indexPath;
  }

  public Class<? extends Analyzer> getAnalyzerClass() {
    return analyzerClass;
  }

  public int getNumberOfMappers() {
    return numberOfMappers;
  }

  public int getNumberOfReducers() {
    return numberOfReducers;
  }

  public float getMaximumDocumentFrequency() {
    return maximumDocumentFrequency;
  }

  public float getMinimumDocumentFrequency() {
    return minimumDocumentFrequency;
  }

  public boolean isLocalMerge() {
    return localMerge;
  }
}