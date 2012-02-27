package cc.mrlda;

import java.io.IOException;
import java.io.StringReader;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
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
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.util.map.HMapII;

public class TokenizeDocument extends Configured implements Tool {
  protected static enum MyCounter {
    TOTAL_DOCS, TOTAL_TERMS,
  }

  public static final String TOKEN = "token";
  public static final String INDEX = "index";

  static final Logger sLogger = Logger.getLogger(TokenizeDocument.class);

  public static class TokenizeMapper extends MapReduceBase implements
      Mapper<LongWritable, Text, Text, PairOfInts> {
    private Text token = new Text();
    private PairOfInts counts = new PairOfInts();

    private OutputCollector<Text, HMapSIW> outputDocument = null;
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
        outputDocument = multipleOutputs.getCollector(Settings.DOCUMENT, Settings.DOCUMENT,
            reporter);
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

    options.addOption(FileMerger.DELETE_SOURCE_OPTION, false, "delete sources after merging");

    String inputPath = null;
    String outputPath = null;
    int numberOfMappers = Settings.DEFAULT_NUMBER_OF_MAPPERS;
    int numberOfReducers = Settings.DEFAULT_NUMBER_OF_REDUCERS;
    boolean localMerge = FileMerger.LOCAL_MERGE;
    boolean deleteSource = FileMerger.DELETE_SOURCE;

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

      if (line.hasOption(FileMerger.DELETE_SOURCE_OPTION)) {
        deleteSource = true;
      }
    } catch (ParseException pe) {
      System.err.println(pe.getMessage());
      formatter.printHelp(VariationalInference.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      System.err.println(nfe.getMessage());
      System.exit(0);
    }

    if (!outputPath.endsWith(Path.SEPARATOR)) {
      outputPath += Path.SEPARATOR;
    }

    run(inputPath, outputPath, numberOfMappers, numberOfReducers, localMerge, deleteSource);

    return 0;
  }

  public Path run(String inputPath, String outputPath, int numberOfMappers, int numberOfReducers,
      boolean localMerge, boolean deleteSource) throws Exception {
    if (!outputPath.endsWith(Path.SEPARATOR)) {
      outputPath += Path.SEPARATOR;
    }

    String tempPath = outputPath + Settings.TEMP;

    sLogger.info("Tool: " + TokenizeDocument.class.getSimpleName());
    sLogger.info(" - input path: " + inputPath);
    sLogger.info(" - output path: " + outputPath);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + numberOfReducers);

    JobConf conf = new JobConf(TokenizeDocument.class);
    conf.setJobName(TokenizeDocument.class.getSimpleName());
    FileSystem fs = FileSystem.get(conf);

    // Delete the output directory if it exists already
    fs.delete(new Path(outputPath), true);

    MultipleOutputs.addMultiNamedOutput(conf, Settings.DOCUMENT, SequenceFileOutputFormat.class,
        Text.class, HMapSIW.class);

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
    FileOutputFormat.setCompressOutput(conf, false);

    long startTime = System.currentTimeMillis();
    RunningJob job = JobClient.runJob(conf);
    sLogger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0
        + " seconds");

    String documentGlobString = tempPath + Path.SEPARATOR + Settings.DOCUMENT + Settings.STAR;
    String documentString = outputPath + Settings.DOCUMENT;
    Path documentPath = null;

    String tokenGlobString = tempPath + Path.SEPARATOR + "part-" + Settings.STAR;
    String tokenString = outputPath + TOKEN;
    Path tokenPath = null;

    if (localMerge) {
      documentPath = FileMerger.mergeSequenceFiles(documentGlobString, documentString, 0,
          Text.class, HMapSIW.class, deleteSource);
      tokenPath = FileMerger.mergeSequenceFiles(tokenGlobString, tokenString, 0, Text.class,
          PairOfInts.class, deleteSource);
    } else {
      documentPath = FileMerger.mergeSequenceFiles(documentGlobString, documentString,
          numberOfMappers, Text.class, HMapSIW.class, deleteSource);
      tokenPath = FileMerger.mergeSequenceFiles(tokenGlobString, tokenString, numberOfMappers,
          Text.class, PairOfInts.class, deleteSource);
    }

    fs.delete(new Path(tempPath), true);

    SequenceFile.Reader sequenceFileReader = null;
    SequenceFile.Writer sequenceFileWriter = null;
    SequenceFile.Writer sequenceFileLDADocumentWriter = null;

    Path tokenIndexPath = new Path(outputPath + INDEX + Path.SEPARATOR + "token");
    fs.createNewFile(tokenIndexPath);
    Path titleIndexPath = new Path(outputPath + INDEX + Path.SEPARATOR + "title");
    fs.createNewFile(titleIndexPath);
    Path documentIndexPath = new Path(outputPath + INDEX + Path.SEPARATOR + "document");
    fs.createNewFile(documentIndexPath);

    try {
      sequenceFileReader = new SequenceFile.Reader(fs, tokenPath, conf);
      sequenceFileWriter = new SequenceFile.Writer(fs, conf, tokenIndexPath, IntWritable.class,
          Text.class);
      Map<String, Integer> tokenIndex = importTokens(sequenceFileReader, sequenceFileWriter);

      sequenceFileReader = new SequenceFile.Reader(fs, documentPath, conf);
      sequenceFileWriter = new SequenceFile.Writer(fs, conf, titleIndexPath, IntWritable.class,
          Text.class);
      sequenceFileLDADocumentWriter = new SequenceFile.Writer(fs, conf, documentIndexPath,
          IntWritable.class, Text.class);
      exportLDADocument(sequenceFileReader, sequenceFileLDADocumentWriter, sequenceFileWriter,
          tokenIndex);
    } finally {
      IOUtils.closeStream(sequenceFileReader);
      IOUtils.closeStream(sequenceFileWriter);
    }

    Counters counters = job.getCounters();
    int documentOldCount = (int) counters.findCounter(MyCounter.TOTAL_DOCS).getCounter();
    sLogger.info("Total number of documents is: " + documentOldCount);

    int termCount = (int) counters.findCounter(MyCounter.TOTAL_TERMS).getCounter();
    sLogger.info("Total number of terms is: " + termCount);

    return documentIndexPath;
  }

  public static Map<String, Integer> importTokens(SequenceFile.Reader sequenceFileReader,
      SequenceFile.Writer sequenceFileWriter) throws IOException {
    Map<String, Integer> tokenIndex = new HashMap<String, Integer>();
    int index = 0;

    Text text = new Text();
    PairOfInts pairOfInts = new PairOfInts();
    IntWritable intWritable = new IntWritable();

    while (sequenceFileReader.next(text, pairOfInts)) {
      index++;
      tokenIndex.put(text.toString(), index);

      intWritable.set(index);
      sequenceFileWriter.append(intWritable, text);
    }

    return tokenIndex;
  }

  public static void exportLDADocument(SequenceFile.Reader sequenceFileReader,
      SequenceFile.Writer sequenceLDADocumentWriter, SequenceFile.Writer sequenceTitleWriter,
      Map<String, Integer> tokenIndex) throws IOException {
    Text text = new Text();
    HMapSIW map = new HMapSIW();

    Iterator<String> itr = null;
    String temp = null;

    HMapII context = new HMapII();
    LDADocument ldaDocument = new LDADocument();
    IntWritable title = new IntWritable();
    int index = 0;

    while (sequenceFileReader.next(text, map)) {
      context.clear();
      itr = map.keySet().iterator();
      while (itr.hasNext()) {
        temp = itr.next();
        Preconditions.checkArgument(tokenIndex.containsKey(temp), "How surprise? Token " + temp
            + " not found in the index file...");
        context.put(tokenIndex.get(temp), map.get(temp));
      }
      ldaDocument.setDocument(context);

      index++;
      title.set(index);
      sequenceTitleWriter.append(title, text);
      sequenceLDADocumentWriter.append(title, context);
    }
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new TokenizeDocument(), args);
    System.exit(res);
  }
}