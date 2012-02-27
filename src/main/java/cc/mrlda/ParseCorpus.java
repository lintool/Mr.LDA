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

import cc.mrlda.util.FileMerger;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapSIW;
import edu.umd.cloud9.io.pair.PairOfIntString;
import edu.umd.cloud9.io.pair.PairOfInts;

public class ParseCorpus extends Configured implements Tool {
  protected static enum MyCounter {
    TOTAL_DOCS, TOTAL_TERMS,
  }

  public static final String DOCUMENT = "document";
  public static final String TOKEN = "token";
  public static final String INDEX = "index";
  public static final String TITLE = "title";

  static final Logger sLogger = Logger.getLogger(ParseCorpus.class);

  public static class TokenizeMapper extends MapReduceBase implements
      Mapper<LongWritable, Text, Text, PairOfInts> {
    private Text token = new Text();
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
      TermAttribute term = stream.addAttribute(TermAttribute.class);
      while (stream.incrementToken()) {
        docContent.increment(term.term());
      }
      outputTitle.collect(docTitle, NullWritable.get());
      outputDocument.collect(docTitle, docContent);

      itr = docContent.keySet().iterator();
      while (itr.hasNext()) {
        temp = itr.next();
        token.set(temp);
        counts.set(1, docContent.get(temp));
        output.collect(token, counts);
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

  public static class IndexMapper extends MapReduceBase implements
      Mapper<Text, HMapSIW, IntWritable, Document> {
    private static Map<String, Integer> tokenIndex = null;
    private static Map<String, Integer> titleIndex = null;

    IntWritable index = new IntWritable();
    Document document = new Document();

    @SuppressWarnings("deprecation")
    public void map(Text key, HMapSIW value, OutputCollector<IntWritable, Document> output,
        Reporter reporter) throws IOException {

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

              if (path.getName().startsWith(TOKEN)) {
                Preconditions.checkArgument(tokenIndex == null,
                    "Token index was initialized already...");
                tokenIndex = ParseCorpus.importParameter(sequenceFileReader);
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

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(Settings.HELP_OPTION, false, "print the help message");
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input file or directory").create(Settings.INPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("output directory").create(Settings.OUTPUT_OPTION));
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
    options.addOption(FileMerger.LOCAL_MERGE_OPTION, false,
        "merge output files and parameters locally");

    String inputPath = null;
    String outputPath = null;
    int numberOfMappers = Settings.DEFAULT_NUMBER_OF_MAPPERS;
    int numberOfReducers = Settings.DEFAULT_NUMBER_OF_REDUCERS;
    boolean localMerge = FileMerger.LOCAL_MERGE;

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

      if (line.hasOption(FileMerger.LOCAL_MERGE_OPTION)) {
        localMerge = true;
      }
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

    run(inputPath, outputPath, numberOfMappers, numberOfReducers, localMerge);

    return 0;
  }

  public Path run(String inputPath, String outputPath, int numberOfMappers, int numberOfReducers,
      boolean localMerge) throws Exception {
    if (!outputPath.endsWith(Path.SEPARATOR)) {
      outputPath += Path.SEPARATOR;
    }

    String tempPath = outputPath + Settings.TEMP;

    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName());
    sLogger.info(" - input path: " + inputPath);
    sLogger.info(" - output path: " + outputPath);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + numberOfReducers);

    JobConf conf = new JobConf(ParseCorpus.class);
    conf.setJobName(ParseCorpus.class.getSimpleName() + " - tokenize");
    FileSystem fs = FileSystem.get(conf);

    // Delete the output directory if it exists already
    fs.delete(new Path(outputPath), true);

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
    FileOutputFormat.setOutputPath(conf, new Path(tempPath));
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

    String titleGlobString = tempPath + Path.SEPARATOR + TITLE + Settings.STAR;
    String titleString = outputPath + TITLE;
    Path titlePath = null;

    String tokenGlobString = tempPath + Path.SEPARATOR + "part-" + Settings.STAR;
    String tokenString = outputPath + TOKEN;
    Path tokenPath = null;

    if (localMerge) {
      titlePath = FileMerger.mergeSequenceFiles(titleGlobString, titleString, 0, Text.class,
          NullWritable.class, true);
      tokenPath = FileMerger.mergeSequenceFiles(tokenGlobString, tokenString, 0, Text.class,
          PairOfInts.class, true);
    } else {
      titlePath = FileMerger.mergeSequenceFiles(titleGlobString, titleString, numberOfMappers,
          Text.class, NullWritable.class, true);
      tokenPath = FileMerger.mergeSequenceFiles(tokenGlobString, tokenString, numberOfMappers,
          Text.class, PairOfInts.class, true);
    }

    SequenceFile.Reader sequenceFileReader = null;
    SequenceFile.Writer sequenceFileWriter = null;

    Path tokenIndexPath = new Path(outputPath + INDEX + Path.SEPARATOR + TOKEN);
    fs.createNewFile(tokenIndexPath);
    Path titleIndexPath = new Path(outputPath + INDEX + Path.SEPARATOR + TITLE);
    fs.createNewFile(titleIndexPath);

    try {
      sequenceFileReader = new SequenceFile.Reader(fs, tokenPath, conf);
      sequenceFileWriter = new SequenceFile.Writer(fs, conf, tokenIndexPath, IntWritable.class,
          Text.class);
      int tokenCounts = exportTokens(sequenceFileReader, sequenceFileWriter);
      Preconditions.checkArgument(tokenCounts == termCount,
          "How embarrassing, mismatch happened for token indices...");
      sLogger.info("Successfully index all the tokens to " + tokenIndexPath);

      sequenceFileReader = new SequenceFile.Reader(fs, titlePath, conf);
      sequenceFileWriter = new SequenceFile.Writer(fs, conf, titleIndexPath, IntWritable.class,
          Text.class);
      int titleCount = exportTitles(sequenceFileReader, sequenceFileWriter);
      Preconditions.checkArgument(titleCount == documentCount,
          "How embarrassing, mismatch happened for title indices...");
      sLogger.info("Successfully index all the titles to " + titleIndexPath);
    } finally {
      IOUtils.closeStream(sequenceFileReader);
      IOUtils.closeStream(sequenceFileWriter);
    }

    // JobConf conf = new JobConf(IndexDocument.class);
    conf.clear();
    conf.setJobName(ParseCorpus.class.getSimpleName() + " - index");
    fs = FileSystem.get(conf);

    Preconditions.checkArgument(fs.exists(tokenIndexPath), "Missing token index files...");
    DistributedCache.addCacheFile(tokenIndexPath.toUri(), conf);
    Preconditions.checkArgument(fs.exists(titleIndexPath), "Missing title index files...");
    DistributedCache.addCacheFile(titleIndexPath.toUri(), conf);

    conf.setNumMapTasks(numberOfMappers);
    conf.setNumReduceTasks(0);
    conf.setMapperClass(TokenizeMapper.class);

    conf.setMapOutputKeyClass(IntWritable.class);
    conf.setMapOutputValueClass(Document.class);
    conf.setOutputKeyClass(IntWritable.class);
    conf.setOutputValueClass(Document.class);

    conf.setInputFormat(SequenceFileInputFormat.class);
    conf.setOutputFormat(SequenceFileOutputFormat.class);

    String documentGlobString = tempPath + Path.SEPARATOR + DOCUMENT + Settings.STAR;
    String documentString = outputPath + DOCUMENT;
    Path documentPath = new Path(documentString);

    FileInputFormat.setInputPaths(conf, new Path(documentGlobString));
    FileOutputFormat.setOutputPath(conf, documentPath);
    FileOutputFormat.setCompressOutput(conf, true);

    try {
      startTime = System.currentTimeMillis();
      job = JobClient.runJob(conf);
      sLogger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0
          + " seconds");
    } finally {
      fs.delete(new Path(tempPath), true);
    }

    return documentPath;
  }

  public static int exportTokens(SequenceFile.Reader sequenceFileReader,
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
    int res = ToolRunner.run(new Configuration(), new ParseCorpus(), args);
    System.exit(res);
  }
}