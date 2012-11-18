package cc.mrlda.polylda;

import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.StringTokenizer;

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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextInputFormat;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.mapred.lib.NullOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.FileMerger;
import edu.umd.cloud9.io.array.ArrayListWritable;
import edu.umd.cloud9.io.map.HMapSIW;
import edu.umd.cloud9.io.pair.PairOfIntString;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.io.triple.TripleOfInts;
import edu.umd.cloud9.util.map.HMapII;

public class ParseCorpus extends Configured implements Tool {
  static final Logger sLogger = Logger.getLogger(ParseCorpus.class);

  public static final String DOCUMENT = "document";
  public static final String TERM = "term";
  public static final String TITLE = "title";
  public static final String INDEX = "index";

  public static final String MINIMUM_DOCUMENT_FREQUENCY = "minimumdocumentfrequency";
  public static final String MAXIMUM_DOCUMENT_FREQUENCY = "maximumdocumentfrequency";
  public static final String MINIMUM_TERM_FREQUENCY = "minimumtermfrequency";
  public static final String MAXIMUM_TERM_FREQUENCY = "maximumtermfrequency";

  public static final float DEFAULT_MINIMUM_DOCUMENT_FREQUENCY = 0.0f;
  public static final float DEFAULT_MAXIMUM_DOCUMENT_FREQUENCY = 1.0f;
  public static final float DEFAULT_MINIMUM_TERM_FREQUENCY = 0.0f;
  public static final float DEFAULT_MAXIMUM_TERM_FREQUENCY = 1.0f;

  private static enum MyCounter {
    TOTAL_TYPES, // total number of distinct words in all documents across all languages
    TOTAL_TERMS, // total number of words in all documents across all languages
    TOTAL_DOCS, // total number of write-ups appeared in the corpus
    TOTAL_ARTICLES, // total number of write-ups appeared in the corpus across all languages
    TOTAL_COLLAPSED_DOCS, // total number of collapsed documents during indexing
  }

  // TODO: change the counter names
  public static final String TOTAL_DOCS_IN_LANGUAGE = "Total Documents in Language";
  public static final String LEFT_OVER_DOCS_IN_LANGUAGE = "Left-over Documents in Language";
  public static final String COLLAPSED_DOCS_IN_LANGUAGE = "Collapsed Documents in Language";
  public static final String TOTAL_DOCS_WITH_MISSING_LANGUAGES = "Total Documents with Missing Languages";

  public static final String TOTAL_TERMS_IN_LANGUAGE = "Total Terms in Language";
  public static final String LOW_DOCUMENT_FREQUENCY_TERMS_IN_LANGUAGE = "Low DF Terms in Language";
  public static final String HIGH_DOCUMENT_FREQUENCY_TERMS_IN_LANGUAGE = "High DF Terms in Language";
  public static final String LEFT_OVER_TERMS_IN_LANGUAGE = "Left-over Terms in Language";

  public static final String NULL = "null";

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(Settings.HELP_OPTION, false, "print the help message");
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input file(s) or directory").isRequired().create(Settings.INPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("output directory").isRequired().create(Settings.OUTPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("number of languages").create(Settings.LANGUAGE_OPTION));
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

    String inputPath = null;
    String outputPath = null;
    int numberOfLanguages = 0;
    int numberOfMappers = Settings.DEFAULT_NUMBER_OF_MAPPERS;
    int numberOfReducers = Settings.DEFAULT_NUMBER_OF_REDUCERS;
    boolean localMerge = FileMerger.LOCAL_MERGE;
    float maximumDocumentFrequency = DEFAULT_MAXIMUM_DOCUMENT_FREQUENCY;
    float minimumDocumentFrequency = DEFAULT_MINIMUM_DOCUMENT_FREQUENCY;
    
    Configuration configuration = this.getConf();
    CommandLineParser parser = new GnuParser();
    HelpFormatter formatter = new HelpFormatter();
    try {
      CommandLine line = parser.parse(options, args);

      if (line.hasOption(Settings.HELP_OPTION)) {
        ToolRunner.printGenericCommandUsage(System.out);
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

      if (line.hasOption(Settings.LANGUAGE_OPTION)) {
        numberOfLanguages = Integer.parseInt(line.getOptionValue(Settings.LANGUAGE_OPTION));
      } else {
        throw new ParseException("Parsing failed due to " + Settings.LANGUAGE_OPTION
            + " not initialized...");
      }
      Preconditions.checkArgument(numberOfLanguages > 0, "Illegal settings for "
          + Settings.LANGUAGE_OPTION + " option: must be strictly positive...");

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
      System.err.println(pe.getMessage());
      ToolRunner.printGenericCommandUsage(System.err);
      formatter.printHelp(ParseCorpus.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      System.err.println(nfe.getMessage());
      System.exit(0);
    }

    if (!outputPath.endsWith(Path.SEPARATOR)) {
      outputPath += Path.SEPARATOR;
    }
    String indexPath = outputPath + INDEX;

    // Delete the output directory if it exists already
    FileSystem fs = FileSystem.get(new JobConf(configuration, ParseCorpus.class));
    fs.delete(new Path(outputPath), true);

    try {
      int[][] corpusStatistics = tokenizeDocument(configuration, inputPath, indexPath, numberOfLanguages,
          numberOfMappers, numberOfReducers);
      int[] documentCount = corpusStatistics[0];
      int[] termCount = corpusStatistics[1];
      Preconditions.checkArgument(documentCount.length == numberOfLanguages + 1,
          "Unexpected document counts array...");
      Preconditions.checkArgument(termCount.length == numberOfLanguages + 1,
          "Unexpected term counts array...");

      float[] minimumDocumentCount = new float[documentCount.length];
      float[] maximumDocumentCount = new float[documentCount.length];
      for (int languageIndex = 0; languageIndex < documentCount.length; languageIndex++) {
        minimumDocumentCount[languageIndex] = documentCount[languageIndex]
            * minimumDocumentFrequency;
        maximumDocumentCount[languageIndex] = documentCount[languageIndex]
            * maximumDocumentFrequency;
      }

      String titleGlobString = indexPath + Path.SEPARATOR + TITLE + Settings.UNDER_SCORE + TITLE
          + Settings.DASH + Settings.STAR;
      String titleString = outputPath + TITLE;
      // Path titleIndexPath = indexTitle(titleGlobString, titleString, numberOfMappers);
      Path titleIndexPath = null;
      if (localMerge) {
        titleIndexPath = indexTitle(configuration, titleGlobString, titleString, 0);
      } else {
        titleIndexPath = indexTitle(configuration, titleGlobString, titleString, numberOfMappers);
      }

      String termGlobString = indexPath + Path.SEPARATOR + "part-" + Settings.STAR;
      String termString = outputPath + TERM;
      Path[] termIndexPath = indexTerm(configuration, termGlobString, termString, numberOfLanguages,
          numberOfMappers, minimumDocumentCount, maximumDocumentCount);

      String documentGlobString = indexPath + Path.SEPARATOR + DOCUMENT + Settings.UNDER_SCORE
          + DOCUMENT + Settings.DASH + Settings.STAR;
      String documentString = outputPath + DOCUMENT;
      Path documentPath = indexDocument(configuration, documentGlobString, documentString, termString,
          titleString, numberOfLanguages, numberOfMappers);
    } finally {
      fs.delete(new Path(indexPath), true);
    }

    return 0;
  }

  public static class TokenizeMapper extends MapReduceBase implements
      Mapper<LongWritable, Text, PairOfIntString, PairOfInts> {
    private PairOfIntString term = new PairOfIntString();
    private PairOfInts counts = new PairOfInts();

    private int numberOfLanguages = 0;

    private OutputCollector<Text, ArrayListWritable<HMapSIW>> outputDocument = null;
    private OutputCollector<Text, NullWritable> outputTitle = null;
    private MultipleOutputs multipleOutputs = null;

    // TODO: make this analyzer as an option to the user
    // private static final StandardAnalyzer standardAnalyzer = new
    // StandardAnalyzer(Version.LUCENE_35);
    // private TokenStream stream = null;

    private Text docTitle = new Text();
    private ArrayListWritable<HMapSIW> docContent = new ArrayListWritable<HMapSIW>();
    private HMapSIW docLanguageContent = null;
    private Iterator<String> itr = null;
    private String temp = null;

    private StringTokenizer stringTokenizer1 = null;
    private StringTokenizer stringTokenizer2 = null;

    @SuppressWarnings("deprecation")
    public void map(LongWritable key, Text value,
        OutputCollector<PairOfIntString, PairOfInts> output, Reporter reporter) throws IOException {
      if (outputDocument == null) {
        outputDocument = multipleOutputs.getCollector(DOCUMENT, DOCUMENT, reporter);
        outputTitle = multipleOutputs.getCollector(TITLE, TITLE, reporter);
      }

      stringTokenizer1 = new StringTokenizer(value.toString(), "" + Settings.TAB);
      Preconditions.checkArgument(stringTokenizer1.countTokens() == numberOfLanguages + 1,
          "Illegal settings for " + Settings.LANGUAGE_OPTION
              + " option: it does not agree with the corpus...");

      int missingLanguages = 0;
      docTitle.set(stringTokenizer1.nextToken().trim());
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        docLanguageContent = docContent.get(languageIndex);
        docLanguageContent.clear();
        temp = stringTokenizer1.nextToken();
        if (temp.equalsIgnoreCase(NULL)) {
          missingLanguages++;
          continue;
        }

        stringTokenizer2 = new StringTokenizer(temp);
        while (stringTokenizer2.hasMoreTokens()) {
          docLanguageContent.increment(stringTokenizer2.nextToken());
        }

        reporter.incrCounter(MyCounter.TOTAL_ARTICLES, 1);
        reporter.incrCounter(TOTAL_DOCS_IN_LANGUAGE, "" + (languageIndex + 1), 1);
      }

      reporter.incrCounter(TOTAL_DOCS_WITH_MISSING_LANGUAGES, "" + missingLanguages, 1);

      // TODO: enable using different analyzer
      // temp = value.toString();
      // int index = temp.indexOf(Settings.TAB);
      // docTitle.set(temp.substring(0, index).trim());
      // docContent = new HMapSIW();
      // stream = standardAnalyzer.tokenStream("contents,",
      // new StringReader(temp.substring(index + 1)));
      // TermAttribute termAttribute = stream.addAttribute(TermAttribute.class);
      // while (stream.incrementToken()) {
      // docContent.increment(termAttribute.term());
      // }

      outputTitle.collect(docTitle, NullWritable.get());
      outputDocument.collect(docTitle, docContent);

      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        itr = docContent.get(languageIndex).keySet().iterator();
        while (itr.hasNext()) {
          temp = itr.next();
          term.set(languageIndex + 1, temp);
          counts.set(1, docContent.get(languageIndex).get(temp));
          output.collect(term, counts);
        }
      }

      reporter.incrCounter(MyCounter.TOTAL_DOCS, 1);
    }

    public void configure(JobConf conf) {
      multipleOutputs = new MultipleOutputs(conf);
      numberOfLanguages = conf.getInt(Settings.PROPERTY_PREFIX + "model.languages", 0);

      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        docContent.add(new HMapSIW());
      }
    }

    public void close() throws IOException {
      multipleOutputs.close();
    }
  }

  public static class TokenizeCombiner extends MapReduceBase implements
      Reducer<PairOfIntString, PairOfInts, PairOfIntString, PairOfInts> {
    private PairOfInts counts = new PairOfInts();

    public void reduce(PairOfIntString key, Iterator<PairOfInts> values,
        OutputCollector<PairOfIntString, PairOfInts> output, Reporter reporter) throws IOException {
      int documentFrequency = 0;
      int termFrequency = 0;

      while (values.hasNext()) {
        counts = values.next();
        documentFrequency += counts.getLeftElement();
        termFrequency += counts.getRightElement();
      }

      counts.set(documentFrequency, termFrequency);
      output.collect(key, counts);
    }
  }

  public static class TokenizeReducer extends MapReduceBase implements
      Reducer<PairOfIntString, PairOfInts, PairOfIntString, PairOfInts> {
    private PairOfInts counts = new PairOfInts();

    public void reduce(PairOfIntString key, Iterator<PairOfInts> values,
        OutputCollector<PairOfIntString, PairOfInts> output, Reporter reporter) throws IOException {
      int documentFrequency = 0;
      int termFrequency = 0;

      while (values.hasNext()) {
        counts = values.next();
        documentFrequency += counts.getLeftElement();
        termFrequency += counts.getRightElement();
      }

      counts.set(documentFrequency, termFrequency);
      output.collect(key, counts);

      reporter.incrCounter(TOTAL_TERMS_IN_LANGUAGE, "" + key.getLeftElement(), 1);
      reporter.incrCounter(MyCounter.TOTAL_TERMS, termFrequency);
      reporter.incrCounter(MyCounter.TOTAL_TYPES, 1);
    }
  }

  public int[][] tokenizeDocument(Configuration configuration, String inputPath, String outputPath, int numberOfLanguages,
      int numberOfMappers, int numberOfReducers) throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName() + " - tokenize document");
    sLogger.info(" - input path: " + inputPath);
    sLogger.info(" - output path: " + outputPath);
    sLogger.info(" - number of languages: " + numberOfLanguages);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + numberOfReducers);

    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    conf.setJobName(ParseCorpus.class.getSimpleName() + " - tokenize document");
    FileSystem fs = FileSystem.get(conf);

    MultipleOutputs.addMultiNamedOutput(conf, DOCUMENT, SequenceFileOutputFormat.class, Text.class,
        ArrayListWritable.class);
    MultipleOutputs.addMultiNamedOutput(conf, TITLE, SequenceFileOutputFormat.class, Text.class,
        NullWritable.class);

    conf.setInt(Settings.PROPERTY_PREFIX + "model.languages", numberOfLanguages);

    conf.setNumMapTasks(numberOfMappers);
    conf.setNumReduceTasks(numberOfReducers);

    conf.setMapperClass(TokenizeMapper.class);
    conf.setReducerClass(TokenizeReducer.class);
    conf.setCombinerClass(TokenizeCombiner.class);

    conf.setMapOutputKeyClass(PairOfIntString.class);
    conf.setMapOutputValueClass(PairOfInts.class);
    conf.setOutputKeyClass(PairOfIntString.class);
    conf.setOutputValueClass(PairOfInts.class);

    conf.setInputFormat(TextInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    FileInputFormat.setInputPaths(conf, new Path(inputPath));
    FileOutputFormat.setOutputPath(conf, new Path(outputPath));
    FileOutputFormat.setCompressOutput(conf, true);

    long startTime = System.currentTimeMillis();
    RunningJob job = JobClient.runJob(conf);
    sLogger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0
        + " seconds");

    Counters counters = job.getCounters();

    int articleCount = (int) counters.findCounter(MyCounter.TOTAL_ARTICLES).getCounter();
    sLogger.info("Total number of articles across all languages is: " + articleCount);
    int[] documentCount = new int[numberOfLanguages + 1];
    documentCount[0] = (int) counters.findCounter(MyCounter.TOTAL_DOCS).getCounter();
    sLogger.info("Total number of documents is: " + documentCount[0]);
    for (int languageIndex = 1; languageIndex <= numberOfLanguages; languageIndex++) {
      documentCount[languageIndex] = (int) counters.findCounter(TOTAL_DOCS_IN_LANGUAGE,
          "" + languageIndex).getCounter();
      sLogger.info("Total number of terms in language " + languageIndex + " is: "
          + documentCount[languageIndex]);
    }

    int[] termCount = new int[numberOfLanguages + 1];
    termCount[0] = (int) counters.findCounter(MyCounter.TOTAL_TYPES).getCounter();
    sLogger.info("Total number of terms is: " + termCount[0]);

    for (int languageIndex = 1; languageIndex <= numberOfLanguages; languageIndex++) {
      termCount[languageIndex] = (int) counters.findCounter(TOTAL_TERMS_IN_LANGUAGE,
          "" + languageIndex).getCounter();
      sLogger.info("Total number of terms in language " + languageIndex + " is: "
          + termCount[languageIndex]);
    }

    int[][] corpusStatistics = new int[2][numberOfLanguages + 1];
    corpusStatistics[0] = documentCount;
    corpusStatistics[1] = termCount;
    return corpusStatistics;
  }

  public Path indexTitle(Configuration configuration, String inputTitles, String outputTitle, int numberOfMappers)
      throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName() + " - index title");
    sLogger.info(" - input path: " + inputTitles);
    sLogger.info(" - output path: " + outputTitle);
    sLogger.info(" - number of mappers: " + numberOfMappers);

    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    FileSystem fs = FileSystem.get(conf);

    Path titleIndexPath = new Path(outputTitle);

    String outputTitleFile = titleIndexPath.getParent() + Path.SEPARATOR + Settings.TEMP
        + FileMerger.generateRandomString();

    // TODO: filemerger local merge can not deal with NullWritable class
    Path titlePath = FileMerger.mergeSequenceFiles(configuration, inputTitles, outputTitleFile, numberOfMappers,
        Text.class, NullWritable.class, true);

    SequenceFile.Reader sequenceFileReader = null;
    SequenceFile.Writer sequenceFileWriter = null;
    fs.createNewFile(titleIndexPath);
    try {
      sequenceFileReader = new SequenceFile.Reader(fs, titlePath, conf);
      sequenceFileWriter = new SequenceFile.Writer(fs, conf, titleIndexPath, IntWritable.class,
          Text.class);
      cc.mrlda.ParseCorpus.exportTitles(sequenceFileReader, sequenceFileWriter);
      sLogger.info("Successfully index all the titles to " + titleIndexPath);
    } finally {
      IOUtils.closeStream(sequenceFileReader);
      IOUtils.closeStream(sequenceFileWriter);
      fs.delete(new Path(outputTitleFile), true);
    }

    return titleIndexPath;
  }

  public static class IndexTermMapper extends MapReduceBase implements
      Mapper<PairOfIntString, PairOfInts, TripleOfInts, Text> {
    private int numberOfLanguages = 0;
    private float[] minimumDocumentCount = null;
    private float[] maximumDocumentCount = null;

    private TripleOfInts outputKey = new TripleOfInts();
    private Text outputValue = new Text();

    @SuppressWarnings("deprecation")
    public void map(PairOfIntString key, PairOfInts value,
        OutputCollector<TripleOfInts, Text> output, Reporter reporter) throws IOException {
      reporter.incrCounter(MyCounter.TOTAL_TYPES, 1);

      int languageIndex = key.getLeftElement();
      reporter.incrCounter(TOTAL_TERMS_IN_LANGUAGE, "" + languageIndex, 1);

      if (value.getLeftElement() < minimumDocumentCount[languageIndex]) {
        reporter.incrCounter(LOW_DOCUMENT_FREQUENCY_TERMS_IN_LANGUAGE, "" + languageIndex, 1);
        return;
      }
      if (value.getLeftElement() > maximumDocumentCount[languageIndex]) {
        reporter.incrCounter(HIGH_DOCUMENT_FREQUENCY_TERMS_IN_LANGUAGE, "" + languageIndex, 1);
        return;
      }

      outputKey.set(languageIndex, -value.getLeftElement(), -value.getRightElement());
      outputValue.set(key.getRightElement());
      output.collect(outputKey, outputValue);
    }

    public void configure(JobConf conf) {
      numberOfLanguages = conf.getInt(Settings.PROPERTY_PREFIX + "model.languages", 0);
      minimumDocumentCount = new float[numberOfLanguages + 1];
      maximumDocumentCount = new float[numberOfLanguages + 1];
      for (int languageIndex = 0; languageIndex < minimumDocumentCount.length; languageIndex++) {
        minimumDocumentCount[languageIndex] = conf.getFloat(Settings.PROPERTY_PREFIX
            + "corpus.minimum.document.count.language" + languageIndex, 0);
        maximumDocumentCount[languageIndex] = conf.getFloat(Settings.PROPERTY_PREFIX
            + "corpus.maximum.document.count.language" + languageIndex, Float.MAX_VALUE);
      }
    }
  }

  public static class IndexTermReducer extends MapReduceBase implements
      Reducer<TripleOfInts, Text, IntWritable, Text> {
    private int numberOfLanguages = 0;
    private int languageIndex = 0;

    private IntWritable intWritable = new IntWritable();
    private int[] index = null;

    private MultipleOutputs multipleOutputs = null;
    private OutputCollector<IntWritable, Text> outputTerm;

    @SuppressWarnings("deprecation")
    public void reduce(TripleOfInts key, Iterator<Text> values,
        OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
      if (languageIndex != key.getLeftElement()) {
        languageIndex = key.getLeftElement();

        outputTerm = multipleOutputs.getCollector(TERM,
            Settings.LANGUAGE_INDICATOR + languageIndex, reporter);
      }

      while (values.hasNext()) {
        index[languageIndex - 1]++;
        intWritable.set(index[languageIndex - 1]);
        outputTerm.collect(intWritable, values.next());
        reporter.incrCounter(LEFT_OVER_TERMS_IN_LANGUAGE, "" + languageIndex, 1);
      }
    }

    public void configure(JobConf conf) {
      multipleOutputs = new MultipleOutputs(conf);
      numberOfLanguages = conf.getInt(Settings.PROPERTY_PREFIX + "model.languages", 0);
      index = new int[numberOfLanguages];
    }

    public void close() throws IOException {
      multipleOutputs.close();
    }
  }

  public Path[] indexTerm(Configuration configuration, String inputTerms, String outputTerm, int numberOfLanguages,
      int numberOfMappers, float[] minimumDocumentCount, float[] maximumDocumentCount)
      throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName() + " - index term");
    sLogger.info(" - input path: " + inputTerms);
    sLogger.info(" - output path: " + outputTerm);
    sLogger.info(" - number of languages: " + numberOfLanguages);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + 1);
    // sLogger.info(" - minimum document count: " + minimumDocumentCount);
    // sLogger.info(" - maximum document count: " + maximumDocumentCount);

    Path inputTermFiles = new Path(inputTerms);
    Path[] outputTermFile = new Path[numberOfLanguages];

    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    FileSystem fs = FileSystem.get(conf);

    conf.setJobName(ParseCorpus.class.getSimpleName() + " - index term");

    conf.setInt(Settings.PROPERTY_PREFIX + "model.languages", numberOfLanguages);
    for (int languageIndex = 0; languageIndex < minimumDocumentCount.length; languageIndex++) {
      conf.setFloat(Settings.PROPERTY_PREFIX + "corpus.minimum.document.count.language"
          + languageIndex, minimumDocumentCount[languageIndex]);
      conf.setFloat(Settings.PROPERTY_PREFIX + "corpus.maximum.document.count.language"
          + languageIndex, maximumDocumentCount[languageIndex]);
    }

    conf.setNumMapTasks(numberOfMappers);
    conf.setNumReduceTasks(1);
    conf.setMapperClass(IndexTermMapper.class);
    conf.setReducerClass(IndexTermReducer.class);

    conf.setMapOutputKeyClass(TripleOfInts.class);
    conf.setMapOutputValueClass(Text.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(Text.class);

    MultipleOutputs.addMultiNamedOutput(conf, TERM, SequenceFileOutputFormat.class,
        IntWritable.class, Text.class);

    conf.setInputFormat(SequenceFileInputFormat.class);
    // suppress the empty files
    conf.setOutputFormat(NullOutputFormat.class);

    String outputString = (new Path(outputTerm)).getParent() + Path.SEPARATOR + Settings.TEMP
        + FileMerger.generateRandomString();
    Path outputPath = new Path(outputString);
    fs.delete(outputPath, true);

    FileInputFormat.setInputPaths(conf, inputTermFiles);
    FileOutputFormat.setOutputPath(conf, outputPath);
    FileOutputFormat.setCompressOutput(conf, true);

    try {
      long startTime = System.currentTimeMillis();
      RunningJob job = JobClient.runJob(conf);
      sLogger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0
          + " seconds");

      Counters counters = job.getCounters();

      Path inputTermFile = null;
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        int lowDocumentFrequencyTerms = (int) counters.findCounter(
            LOW_DOCUMENT_FREQUENCY_TERMS_IN_LANGUAGE, "" + (languageIndex + 1)).getCounter();
        sLogger.info("Removed " + lowDocumentFrequencyTerms + " low frequency terms in language "
            + (languageIndex + 1));

        int highDocumentFrequencyTerms = (int) counters.findCounter(
            HIGH_DOCUMENT_FREQUENCY_TERMS_IN_LANGUAGE, "" + (languageIndex + 1)).getCounter();
        sLogger.info("Removed " + highDocumentFrequencyTerms + " high frequency terms in language "
            + (languageIndex + 1));

        int leftOverTerms = (int) counters.findCounter(LEFT_OVER_TERMS_IN_LANGUAGE,
            "" + (languageIndex + 1)).getCounter();
        sLogger.info("Total number of left-over terms in language " + (languageIndex + 1) + ": "
            + leftOverTerms);

        inputTermFile = new Path(outputString + Path.SEPARATOR + TERM + Settings.UNDER_SCORE
            + Settings.LANGUAGE_INDICATOR + (languageIndex + 1) + "-r-00000");
        Preconditions.checkArgument(fs.exists(inputTermFile), "Vocabulary suppressed for language "
            + (languageIndex + 1) + "...");
        outputTermFile[languageIndex] = new Path(outputTerm + Settings.UNDER_SCORE
            + Settings.LANGUAGE_INDICATOR + (languageIndex + 1));
        fs.rename(inputTermFile, outputTermFile[languageIndex]);

        sLogger.info("Successfully index all the terms for language " + (languageIndex + 1)
            + " at " + outputTermFile[languageIndex]);
      }
    } finally {
      fs.delete(outputPath, true);
    }

    return outputTermFile;
  }

  public static class IndexDocumentMapper extends MapReduceBase implements
      Mapper<Text, ArrayListWritable<HMapSIW>, IntWritable, Document> {
    private int numberOfLanguages = 0;

    private static Map<String, Integer>[] termIndex = null;
    private static Map<String, Integer> titleIndex = null;

    private IntWritable index = new IntWritable();
    private Document document = new Document();
    private HMapII[] content = null;

    private Iterator<String> itr = null;
    private String temp = null;

    @SuppressWarnings("deprecation")
    public void map(Text key, ArrayListWritable<HMapSIW> value,
        OutputCollector<IntWritable, Document> output, Reporter reporter) throws IOException {

      int missingLanguages = 0;
      boolean collapsedDocument = true;
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        content[languageIndex].clear();

        if (value.get(languageIndex).isEmpty()) {
          missingLanguages++;
          continue;
        }

        reporter.incrCounter(TOTAL_DOCS_IN_LANGUAGE, "" + (languageIndex + 1), 1);

        itr = value.get(languageIndex).keySet().iterator();
        while (itr.hasNext()) {
          temp = itr.next();
          if (termIndex[languageIndex].containsKey(temp)) {
            content[languageIndex].put(termIndex[languageIndex].get(temp), value.get(languageIndex)
                .get(temp));
          }
        }

        if (content[languageIndex].size() == 0) {
          reporter.incrCounter(COLLAPSED_DOCS_IN_LANGUAGE, "" + (languageIndex + 1), 1);
          missingLanguages++;
        } else {
          collapsedDocument = false;
          reporter.incrCounter(LEFT_OVER_DOCS_IN_LANGUAGE, "" + (languageIndex + 1), 1);
          reporter.incrCounter(MyCounter.TOTAL_ARTICLES, 1);
        }
      }
      reporter.incrCounter(TOTAL_DOCS_WITH_MISSING_LANGUAGES, "" + missingLanguages, 1);

      if (collapsedDocument) {
        reporter.incrCounter(MyCounter.TOTAL_COLLAPSED_DOCS, 1);
        return;
      }

      Preconditions.checkArgument(titleIndex.containsKey(key.toString()),
          "How embarrassing! Could not find title " + temp + " in index...");
      index.set(titleIndex.get(key.toString()));

      document.setDocument(content);
      output.collect(index, document);

      reporter.incrCounter(MyCounter.TOTAL_DOCS, 1);
    }

    public void configure(JobConf conf) {
      numberOfLanguages = conf.getInt(Settings.PROPERTY_PREFIX + "model.languages", 0);
      termIndex = new Map[numberOfLanguages];
      content = new HMapII[numberOfLanguages];
      for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
        content[languageIndex] = new HMapII();
      }

      SequenceFile.Reader sequenceFileReader = null;
      try {
        Path[] inputFiles = DistributedCache.getLocalCacheFiles(conf);
        // TODO: check for the missing columns...
        if (inputFiles != null) {
          for (Path path : inputFiles) {
            try {
              sequenceFileReader = new SequenceFile.Reader(FileSystem.getLocal(conf), path, conf);

              if (path.getName().startsWith(TERM)) {
                int languageIndex = Integer.parseInt(path.getName().substring(
                    path.getName().indexOf(Settings.LANGUAGE_INDICATOR)
                        + Settings.LANGUAGE_INDICATOR.length()));
                Preconditions.checkArgument(termIndex[languageIndex - 1] == null,
                    "Term index was initialized already...");
                termIndex[languageIndex - 1] = cc.mrlda.ParseCorpus
                    .importParameter(sequenceFileReader);
              }
              if (path.getName().startsWith(TITLE)) {
                Preconditions.checkArgument(titleIndex == null,
                    "Title index was initialized already...");
                titleIndex = cc.mrlda.ParseCorpus.importParameter(sequenceFileReader);
              } else {
                throw new IllegalArgumentException("Unexpected file in distributed cache: "
                    + path.getName());
              }
            } catch (IllegalArgumentException iae) {
              iae.printStackTrace();
            } catch (IOException ioe) {
              ioe.printStackTrace();
            }
          }
        }
      } catch (IOException ioe) {
        ioe.printStackTrace();
      } finally {
        IOUtils.closeStream(sequenceFileReader);
      }
    }
  }

  public Path indexDocument(Configuration configuration, String inputDocument, String outputDocument, String termIndex,
      String titleIndex, int numberOfLanguages, int numberOfMappers) throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName() + " - index document");
    sLogger.info(" - input path: " + inputDocument);
    sLogger.info(" - output path: " + outputDocument);
    sLogger.info(" - term index path: " + termIndex);
    sLogger.info(" - title index path: " + titleIndex);
    sLogger.info(" - number of languages: " + numberOfLanguages);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + 0);

    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    FileSystem fs = FileSystem.get(conf);

    conf.setJobName(ParseCorpus.class.getSimpleName() + " - index document");

    Path inputDocumentFiles = new Path(inputDocument);
    Path outputDocumentFiles = new Path(outputDocument);

    Path titleIndexPath = new Path(titleIndex);
    Preconditions.checkArgument(fs.exists(titleIndexPath), "Missing title index files...");
    DistributedCache.addCacheFile(titleIndexPath.toUri(), conf);

    for (int languageIndex = 0; languageIndex < numberOfLanguages; languageIndex++) {
      Path termIndexPath = new Path(termIndex + Settings.UNDER_SCORE + Settings.LANGUAGE_INDICATOR
          + (languageIndex + 1));
      Preconditions.checkArgument(fs.exists(termIndexPath),
          "Missing term index files for language " + (languageIndex + 1) + "...");
      DistributedCache.addCacheFile(termIndexPath.toUri(), conf);
    }

    conf.setInt(Settings.PROPERTY_PREFIX + "model.languages", numberOfLanguages);

    conf.setNumMapTasks(numberOfMappers);
    conf.setNumReduceTasks(0);
    conf.setMapperClass(IndexDocumentMapper.class);

    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(Document.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(Document.class);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    FileInputFormat.setInputPaths(conf, inputDocumentFiles);
    FileOutputFormat.setOutputPath(conf, outputDocumentFiles);
    FileOutputFormat.setCompressOutput(conf, false);

    long startTime = System.currentTimeMillis();
    RunningJob job = JobClient.runJob(conf);
    sLogger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0
        + " seconds");
    sLogger.info("Successfully index all the documents at " + outputDocumentFiles);

    Counters counters = job.getCounters();

    int articleCount = (int) counters.findCounter(MyCounter.TOTAL_ARTICLES).getCounter();
    sLogger.info("Total number of articles across all languages is: " + articleCount);
    int collapsedDocumentCount = (int) counters.findCounter(MyCounter.TOTAL_COLLAPSED_DOCS)
        .getCounter();
    sLogger.info("Total number of collapsed documents is: " + collapsedDocumentCount);
    int documentCount = (int) counters.findCounter(MyCounter.TOTAL_DOCS).getCounter();
    sLogger.info("Total number of documents is: " + documentCount);

    return outputDocumentFiles;
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new ParseCorpus(), args);
    System.exit(res);
  }
}