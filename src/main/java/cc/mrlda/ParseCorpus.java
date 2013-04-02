package cc.mrlda;

import java.io.IOException;
import java.io.StringReader;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeSet;

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
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.TermAttribute;
import org.apache.lucene.util.Version;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.FileMerger;
import edu.umd.cloud9.io.map.HMapSIW;
import edu.umd.cloud9.io.pair.PairOfIntString;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.util.map.HMapII;

public class ParseCorpus extends Configured implements Tool {
  static final Logger sLogger = Logger.getLogger(ParseCorpus.class);

  private static enum MyCounter {
    TOTAL_DOCS, TOTAL_TERMS, LOW_DOCUMENT_FREQUENCY_TERMS, HIGH_DOCUMENT_FREQUENCY_TERMS, LEFT_OVER_TERMS, LEFT_OVER_DOCUMENTS, COLLAPSED_DOCUMENTS,
  }

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

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(Settings.HELP_OPTION, false, "print the help message");
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input file(s) or directory").isRequired().create(Settings.INPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("output directory").isRequired().create(Settings.OUTPUT_OPTION));
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
    int numberOfMappers = Settings.DEFAULT_NUMBER_OF_MAPPERS;
    int numberOfReducers = Settings.DEFAULT_NUMBER_OF_REDUCERS;
    float maximumDocumentFrequency = DEFAULT_MAXIMUM_DOCUMENT_FREQUENCY;
    float minimumDocumentFrequency = DEFAULT_MINIMUM_DOCUMENT_FREQUENCY;
    // boolean localMerge = FileMerger.LOCAL_MERGE;

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
    FileSystem fs = FileSystem.get(getConf());
    fs.delete(new Path(outputPath), true);

    try {
      PairOfInts pairOfInts = tokenizeDocument(inputPath, indexPath, numberOfMappers,
          numberOfReducers);
      int totalNumberOfDocuments = pairOfInts.getLeftElement();
      int totalNumberOfTerms = pairOfInts.getRightElement();

      String titleGlobString = indexPath + Path.SEPARATOR + TITLE + Settings.STAR;
      String titleString = outputPath + TITLE;
      Path titleIndexPath = indexTitle(titleGlobString, titleString, numberOfMappers);

      String termGlobString = indexPath + Path.SEPARATOR + "part-" + Settings.STAR;
      String termString = outputPath + TERM;
      Path termIndexPath = indexTerm(termGlobString, termString, numberOfMappers,
          totalNumberOfDocuments * minimumDocumentFrequency, totalNumberOfDocuments
              * maximumDocumentFrequency);

      String documentGlobString = indexPath + Path.SEPARATOR + DOCUMENT + Settings.STAR;
      String documentString = outputPath + DOCUMENT;
      Path documentPath = indexDocument(documentGlobString, documentString,
          termIndexPath.toString(), titleIndexPath.toString(), numberOfMappers);
    } finally {
      fs.delete(new Path(indexPath), true);
    }

    return 0;
  }

  public static class TokenizeMapper extends MapReduceBase implements
      Mapper<LongWritable, Text, Text, PairOfInts> {
    private Text term = new Text();
    private PairOfInts counts = new PairOfInts();

    private OutputCollector<Text, HMapSIW> outputDocument = null;
    private OutputCollector<Text, NullWritable> outputTitle = null;
    private MultipleOutputs multipleOutputs = null;

    private static final StandardAnalyzer standardAnalyzer = new StandardAnalyzer(Version.LUCENE_35);
    private TokenStream stream = null;

    private Text docTitle = new Text();
    private HMapSIW docContent = null;
    private Iterator<String> itr = null;
    private String temp = null;

    @SuppressWarnings("deprecation")
    public void map(LongWritable key, Text value, OutputCollector<Text, PairOfInts> output,
        Reporter reporter) throws IOException {
      if (outputDocument == null) {
        outputDocument = multipleOutputs.getCollector(DOCUMENT, DOCUMENT, reporter);
        outputTitle = multipleOutputs.getCollector(TITLE, TITLE, reporter);
      }

      temp = value.toString();
      int index = temp.indexOf(Settings.TAB);
      docTitle.set(temp.substring(0, index).trim());
      docContent = new HMapSIW();
      stream = standardAnalyzer.tokenStream("contents,",
          new StringReader(temp.substring(index + 1)));
      TermAttribute termAttribute = stream.addAttribute(TermAttribute.class);
      while (stream.incrementToken()) {
        docContent.increment(termAttribute.term());
      }
      outputTitle.collect(docTitle, NullWritable.get());
      outputDocument.collect(docTitle, docContent);

      itr = docContent.keySet().iterator();
      while (itr.hasNext()) {
        temp = itr.next();
        term.set(temp);
        counts.set(1, docContent.get(temp));
        output.collect(term, counts);
      }

      reporter.incrCounter(MyCounter.TOTAL_DOCS, 1);
    }

    public void configure(JobConf conf) {
      multipleOutputs = new MultipleOutputs(conf);
    }

    public void close() throws IOException {
      multipleOutputs.close();
    }
  }

  public static class TokenizeCombiner extends MapReduceBase implements
      Reducer<Text, PairOfInts, Text, PairOfInts> {
    private PairOfInts counts = new PairOfInts();

    public void reduce(Text key, Iterator<PairOfInts> values,
        OutputCollector<Text, PairOfInts> output, Reporter reporter) throws IOException {
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
      Reducer<Text, PairOfInts, Text, PairOfInts> {
    private PairOfInts counts = new PairOfInts();

    public void reduce(Text key, Iterator<PairOfInts> values,
        OutputCollector<Text, PairOfInts> output, Reporter reporter) throws IOException {
      int documentFrequency = 0;
      int termFrequency = 0;

      while (values.hasNext()) {
        counts = values.next();
        documentFrequency += counts.getLeftElement();
        termFrequency += counts.getRightElement();
      }

      counts.set(documentFrequency, termFrequency);
      output.collect(key, counts);

      reporter.incrCounter(MyCounter.TOTAL_TERMS, 1);
    }
  }

  public PairOfInts tokenizeDocument(String inputPath, String outputPath, int numberOfMappers,
      int numberOfReducers) throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName());
    sLogger.info(" - input path: " + inputPath);
    sLogger.info(" - output path: " + outputPath);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + numberOfReducers);

    JobConf conf = new JobConf(getConf());
    conf.setJobName(ParseCorpus.class.getSimpleName() + " - tokenize document");
    FileSystem fs = FileSystem.get(conf);

    MultipleOutputs.addMultiNamedOutput(conf, DOCUMENT, SequenceFileOutputFormat.class, Text.class,
        HMapSIW.class);
    MultipleOutputs.addMultiNamedOutput(conf, TITLE, SequenceFileOutputFormat.class, Text.class,
        NullWritable.class);

    conf.setNumMapTasks(numberOfMappers);
    conf.setNumReduceTasks(numberOfReducers);

    conf.setMapperClass(TokenizeMapper.class);
    conf.setReducerClass(TokenizeReducer.class);
    conf.setCombinerClass(TokenizeCombiner.class);

    conf.setMapOutputKeyClass(Text.class);
    conf.setMapOutputValueClass(PairOfInts.class);
    conf.setOutputKeyClass(Text.class);
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
    int documentCount = (int) counters.findCounter(MyCounter.TOTAL_DOCS).getCounter();
    sLogger.info("Total number of documents is: " + documentCount);

    int termCount = (int) counters.findCounter(MyCounter.TOTAL_TERMS).getCounter();
    sLogger.info("Total number of terms is: " + termCount);

    return new PairOfInts(documentCount, termCount);
  }

  public Path indexTitle(String inputTitles, String outputTitle, int numberOfMappers)
      throws Exception {
    JobConf conf = new JobConf(getConf());
    FileSystem fs = FileSystem.get(conf);

    Path titleIndexPath = new Path(outputTitle);

    String outputTitleFile = titleIndexPath.getParent() + Path.SEPARATOR + Settings.TEMP;
    Path titlePath = FileMerger.mergeSequenceFiles(inputTitles, outputTitleFile, numberOfMappers,
        Text.class, NullWritable.class, true);

    SequenceFile.Reader sequenceFileReader = null;
    SequenceFile.Writer sequenceFileWriter = null;
    fs.createNewFile(titleIndexPath);
    try {
      sequenceFileReader = new SequenceFile.Reader(fs, titlePath, conf);
      sequenceFileWriter = new SequenceFile.Writer(fs, conf, titleIndexPath, IntWritable.class,
          Text.class);
      exportTitles(sequenceFileReader, sequenceFileWriter);
      sLogger.info("Successfully index all the titles to " + titleIndexPath);
    } finally {
      IOUtils.closeStream(sequenceFileReader);
      IOUtils.closeStream(sequenceFileWriter);
      fs.delete(new Path(outputTitleFile), true);
    }

    return titleIndexPath;
  }

  public static class IndexTermMapper extends MapReduceBase implements
      Mapper<Text, PairOfInts, PairOfInts, Text> {
    float minimumDocumentCount = 0;
    float maximumDocumentCount = Float.MAX_VALUE;

    @SuppressWarnings("deprecation")
    public void map(Text key, PairOfInts value, OutputCollector<PairOfInts, Text> output,
        Reporter reporter) throws IOException {
      if (value.getLeftElement() <= minimumDocumentCount) {
        reporter.incrCounter(MyCounter.LOW_DOCUMENT_FREQUENCY_TERMS, 1);
        return;
      }
      if (value.getLeftElement() >= maximumDocumentCount) {
        reporter.incrCounter(MyCounter.HIGH_DOCUMENT_FREQUENCY_TERMS, 1);
        return;
      }
      value.set(-value.getLeftElement(), -value.getRightElement());
      output.collect(value, key);
    }

    public void configure(JobConf conf) {
      minimumDocumentCount = conf.getFloat("corpus.minimum.document.count", 0);
      maximumDocumentCount = conf.getFloat("corpus.maximum.document.count", Float.MAX_VALUE);
    }
  }

  public static class IndexTermReducer extends MapReduceBase implements
      Reducer<PairOfInts, Text, IntWritable, Text> {
    private IntWritable intWritable = new IntWritable();
    private int index = 0;

    @SuppressWarnings("deprecation")
    public void reduce(PairOfInts key, Iterator<Text> values,
        OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
      while (values.hasNext()) {
        index++;
        intWritable.set(index);
        reporter.incrCounter(MyCounter.LEFT_OVER_TERMS, 1);
        output.collect(intWritable, values.next());
      }
    }
  }

  public Path indexTerm(String inputTerms, String outputTerm, int numberOfMappers,
      float minimumDocumentCount, float maximumDocumentCount) throws Exception {
    Path inputTermFiles = new Path(inputTerms);
    Path outputTermFile = new Path(outputTerm);

    JobConf conf = new JobConf(getConf());
    FileSystem fs = FileSystem.get(conf);

    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName());
    sLogger.info(" - input path: " + inputTermFiles);
    sLogger.info(" - output path: " + outputTermFile);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + 1);
    sLogger.info(" - minimum document count: " + minimumDocumentCount);
    sLogger.info(" - maximum document count: " + maximumDocumentCount);

    conf.setJobName(ParseCorpus.class.getSimpleName() + " - index term");

    conf.setNumMapTasks(numberOfMappers);
    conf.setNumReduceTasks(1);
    conf.setMapperClass(IndexTermMapper.class);
    conf.setReducerClass(IndexTermReducer.class);

    conf.setMapOutputKeyClass(PairOfInts.class);
    conf.setMapOutputValueClass(Text.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(Text.class);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    conf.setFloat("corpus.minimum.document.count", minimumDocumentCount);
    conf.setFloat("corpus.maximum.document.count", maximumDocumentCount);

    String outputString = outputTermFile.getParent() + Path.SEPARATOR + Settings.TEMP;
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

      fs.rename(new Path(outputString + Path.SEPARATOR + "part-00000"), outputTermFile);
      sLogger.info("Successfully index all the terms at " + outputTermFile);

      Counters counters = job.getCounters();
      int lowDocumentFrequencyTerms = (int) counters.findCounter(
          MyCounter.LOW_DOCUMENT_FREQUENCY_TERMS).getCounter();
      sLogger.info("Removed " + lowDocumentFrequencyTerms + " low frequency terms.");

      int highDocumentFrequencyTerms = (int) counters.findCounter(
          MyCounter.HIGH_DOCUMENT_FREQUENCY_TERMS).getCounter();
      sLogger.info("Removed " + highDocumentFrequencyTerms + " high frequency terms.");

      int leftOverTerms = (int) counters.findCounter(MyCounter.LEFT_OVER_TERMS).getCounter();
      sLogger.info("Total number of left-over terms: " + leftOverTerms);
    } finally {
      fs.delete(outputPath, true);
    }

    return outputTermFile;
  }

  public static class IndexDocumentMapper extends MapReduceBase implements
      Mapper<Text, HMapSIW, IntWritable, Document> {
    private static Map<String, Integer> termIndex = null;
    private static Map<String, Integer> titleIndex = null;

    private IntWritable index = new IntWritable();
    private Document document = new Document();
    private HMapII content = new HMapII();

    private Iterator<String> itr = null;
    private String temp = null;

    @SuppressWarnings("deprecation")
    public void map(Text key, HMapSIW value, OutputCollector<IntWritable, Document> output,
        Reporter reporter) throws IOException {
      Preconditions.checkArgument(titleIndex.containsKey(key.toString()),
          "How embarrassing! Could not find title " + temp + " in index...");
      content.clear();
      itr = value.keySet().iterator();
      while (itr.hasNext()) {
        temp = itr.next();
        // Preconditions.checkArgument(termIndex.containsKey(temp),
        // "How embarrassing! Could not find term " + temp + " in index...");
        if (termIndex.containsKey(temp)) {
          content.put(termIndex.get(temp), value.get(temp));
        }
      }

      if (content.size() == 0) {
        reporter.incrCounter(MyCounter.COLLAPSED_DOCUMENTS, 1);
        return;
      }

      reporter.incrCounter(MyCounter.LEFT_OVER_DOCUMENTS, 1);
      index.set(titleIndex.get(key.toString()));
      document.setDocument(content);
      output.collect(index, document);
    }

    public void configure(JobConf conf) {
      SequenceFile.Reader sequenceFileReader = null;
      try {
        Path[] inputFiles = DistributedCache.getLocalCacheFiles(conf);
        // TODO: check for the missing columns...
        if (inputFiles != null) {
          for (Path path : inputFiles) {
            try {
              sequenceFileReader = new SequenceFile.Reader(FileSystem.getLocal(conf), path, conf);

              if (path.getName().startsWith(TERM)) {
                Preconditions.checkArgument(termIndex == null,
                    "Term index was initialized already...");
                termIndex = ParseCorpus.importParameter(sequenceFileReader);
              }
              if (path.getName().startsWith(TITLE)) {
                Preconditions.checkArgument(titleIndex == null,
                    "Title index was initialized already...");
                titleIndex = ParseCorpus.importParameter(sequenceFileReader);
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

  public Path indexDocument(String inputDocument, String outputDocument, String termIndex,
      String titleIndex, int numberOfMappers) throws Exception {
    Path inputDocumentFiles = new Path(inputDocument);
    Path outputDocumentFiles = new Path(outputDocument);
    Path termIndexPath = new Path(termIndex);
    Path titleIndexPath = new Path(titleIndex);

    JobConf conf = new JobConf(getConf());
    FileSystem fs = FileSystem.get(conf);

    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName());
    sLogger.info(" - input path: " + inputDocumentFiles);
    sLogger.info(" - output path: " + outputDocumentFiles);
    sLogger.info(" - term index path: " + termIndexPath);
    sLogger.info(" - title index path: " + titleIndexPath);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + 0);

    conf.setJobName(ParseCorpus.class.getSimpleName() + " - index document");

    Preconditions.checkArgument(fs.exists(termIndexPath), "Missing term index files...");
    DistributedCache.addCacheFile(termIndexPath.toUri(), conf);
    Preconditions.checkArgument(fs.exists(titleIndexPath), "Missing title index files...");
    DistributedCache.addCacheFile(titleIndexPath.toUri(), conf);

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
    int collapsedDocuments = (int) counters.findCounter(MyCounter.COLLAPSED_DOCUMENTS).getCounter();
    sLogger.info("Total number of collapsed documnts: " + collapsedDocuments);

    int leftOverDocuments = (int) counters.findCounter(MyCounter.LEFT_OVER_DOCUMENTS).getCounter();
    sLogger.info("Total number of left-over documents: " + leftOverDocuments);

    return outputDocumentFiles;
  }

  public static int exportTitles(SequenceFile.Reader sequenceFileReader,
      SequenceFile.Writer sequenceWriter) throws IOException {
    Text text = new Text();
    IntWritable intWritable = new IntWritable();
    int index = 0;
    while (sequenceFileReader.next(text)) {
      index++;
      intWritable.set(index);
      sequenceWriter.append(intWritable, text);
    }

    return index;
  }

  public static Map<String, Integer> importParameter(SequenceFile.Reader sequenceFileReader)
      throws IOException {
    Map<String, Integer> hashMap = new HashMap<String, Integer>();

    IntWritable intWritable = new IntWritable();
    Text text = new Text();
    while (sequenceFileReader.next(intWritable, text)) {
      hashMap.put(text.toString(), intWritable.get());
    }

    return hashMap;
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new ParseCorpus(), args);
    System.exit(res);
  }

  /**
   * @deprecated
   * @param sequenceFileReader
   * @param sequenceFileWriter
   * @return
   * @throws IOException
   */
  public static int exportTerms(SequenceFile.Reader sequenceFileReader,
      SequenceFile.Writer sequenceFileWriter) throws IOException {
    TreeSet<PairOfIntString> treeMap = new TreeSet<PairOfIntString>(new Comparator() {
      @Override
      public int compare(Object obj1, Object obj2) {
        PairOfIntString entry1 = (PairOfIntString) obj1;
        PairOfIntString entry2 = (PairOfIntString) obj2;
        if (entry1.getLeftElement() > entry2.getLeftElement()) {
          return -1;
        } else if (entry1.getLeftElement() < entry2.getLeftElement()) {
          return entry1.getRightElement().compareTo(entry2.getRightElement());
        } else {
          return 0;
        }
      }
    });

    Text text = new Text();
    PairOfInts pairOfInts = new PairOfInts();
    while (sequenceFileReader.next(text, pairOfInts)) {
      treeMap.add(new PairOfIntString(pairOfInts.getLeftElement(), text.toString()));
    }

    int index = 0;
    IntWritable intWritable = new IntWritable();
    Iterator<PairOfIntString> itr = treeMap.iterator();
    while (itr.hasNext()) {
      index++;
      intWritable.set(index);
      text.set(itr.next().getRightElement());
      sequenceFileWriter.append(intWritable, text);
    }

    return index;
  }
}