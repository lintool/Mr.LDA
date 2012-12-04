package cc.mrlda;

import java.io.BufferedReader;
import java.io.Closeable;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.StringReader;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
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
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.FileMerger;
import edu.umd.cloud9.io.map.HMapSIW;
import edu.umd.cloud9.io.pair.PairOfIntString;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.util.map.HMapII;

public class ParseCorpus extends Configured implements Tool {
  static final Logger sLogger = Logger.getLogger(ParseCorpus.class);

  protected static enum MyCounter {
    TOTAL_DOCS, TOTAL_TERMS, LOW_DOCUMENT_FREQUENCY_TERMS, HIGH_DOCUMENT_FREQUENCY_TERMS, LEFT_OVER_TERMS, LEFT_OVER_DOCUMENTS, COLLAPSED_DOCUMENTS,
  }

  public static final String DOCUMENT = "document";
  public static final String TERM = "term";
  public static final String TITLE = "title";

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    ParseCorpusOptions parseCorpusOptions = new ParseCorpusOptions(args);

    return run(getConf(), parseCorpusOptions);
  }

  private int run(Configuration configuration, ParseCorpusOptions parseCorpusOptions)
      throws Exception {

    String inputPath = parseCorpusOptions.getInputPath();
    String outputPath = parseCorpusOptions.getOutputPath();
    String vocabularyPath = parseCorpusOptions.getIndexPath();
    String stopwordPath = parseCorpusOptions.getStopListPath();
    Class<? extends Analyzer> analyzerClass = parseCorpusOptions.getAnalyzerClass();
    int numberOfMappers = parseCorpusOptions.getNumberOfMappers();
    int numberOfReducers = parseCorpusOptions.getNumberOfReducers();
    float maximumDocumentFrequency = parseCorpusOptions.getMaximumDocumentFrequency();
    float minimumDocumentFrequency = parseCorpusOptions.getMinimumDocumentFrequency();
    boolean localMerge = parseCorpusOptions.isLocalMerge();

    if (!outputPath.endsWith(Path.SEPARATOR)) {
      outputPath += Path.SEPARATOR;
    }
    String indexPath = outputPath + ParseCorpusOptions.INDEX;

    // Delete the output directory if it exists already
    FileSystem fs = FileSystem.get(new JobConf(configuration, ParseCorpus.class));
    fs.delete(new Path(outputPath), true);

    try {
      int[] corpusStatistics = tokenizeDocument(configuration, inputPath, indexPath, stopwordPath,
          analyzerClass, numberOfMappers, numberOfReducers);
      int documentCount = corpusStatistics[0];
      int termsCount = corpusStatistics[1];

      String titleGlobString = indexPath + Path.SEPARATOR + TITLE + Settings.UNDER_SCORE + TITLE
          + Settings.DASH + Settings.STAR;
      String titleString = outputPath + TITLE;

      Path titleIndexPath = null;
      if (localMerge) {
        titleIndexPath = indexTitle(configuration, titleGlobString, titleString, 0);
      } else {
        titleIndexPath = indexTitle(configuration, titleGlobString, titleString, numberOfMappers);
      }

      String termString = outputPath + TERM;
      Path termIndexPath = new Path(termString);
      if (vocabularyPath == null || !fs.exists(new Path(vocabularyPath))) {
        String termGlobString = indexPath + Path.SEPARATOR + "part-" + Settings.STAR;
        termIndexPath = indexTerm(configuration, termGlobString, termString, numberOfMappers,
            documentCount * minimumDocumentFrequency, documentCount * maximumDocumentFrequency);
      } else {
        FileUtil.copy(fs, new Path(vocabularyPath), fs, termIndexPath, false, configuration);
      }

      String documentGlobString = indexPath + Path.SEPARATOR + DOCUMENT + Settings.UNDER_SCORE
          + DOCUMENT + Settings.DASH + Settings.STAR;
      String documentString = outputPath + DOCUMENT;

      Path documentPath = indexDocument(configuration, documentGlobString, documentString,
          termIndexPath.toString(), titleIndexPath.toString(), numberOfMappers);
    } finally {
      fs.delete(new Path(indexPath), true);
    }

    return 0;
  }

  private static class TokenizeMapper extends MapReduceBase implements
      Mapper<LongWritable, Text, Text, PairOfInts> {
    private Text term = new Text();
    private PairOfInts counts = new PairOfInts();

    private OutputCollector<Text, HMapSIW> outputDocument = null;
    private OutputCollector<Text, NullWritable> outputTitle = null;
    private MultipleOutputs multipleOutputs = null;

    private Set<String> stopWordList = null;

    // private static Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_40);
    private Analyzer analyzer = null;
    private TokenStream tokenStream = null;

    private Text docTitle = new Text();
    private HMapSIW docContent = null;
    private Iterator<String> itr = null;
    private String temp = null;
    private String token = null;
    private StringTokenizer stk = null;

    @SuppressWarnings("deprecation")
    public void map(LongWritable key, Text value, OutputCollector<Text, PairOfInts> output,
        Reporter reporter) throws IOException {
      if (outputDocument == null) {
        outputDocument = multipleOutputs.getCollector(DOCUMENT, DOCUMENT, reporter);
        outputTitle = multipleOutputs.getCollector(TITLE, TITLE, reporter);
      }

      temp = value.toString();
      int index = temp.indexOf(Settings.TAB);
      if (index < 0) {
        throw new IndexOutOfBoundsException("Missing title information: " + value.toString());
      }
      docTitle.set(temp.substring(0, index).trim());
      docContent = new HMapSIW();

      if (analyzer == null) {
        stk = new StringTokenizer(temp.substring(index + 1));
        while (stk.hasMoreElements()) {
          token = stk.nextToken();
          if (stopWordList != null && stopWordList.contains(token)) {
            continue;
          }
          docContent.increment(token);
        }
      } else {
        tokenStream = analyzer
            .tokenStream("contents,", new StringReader(temp.substring(index + 1)));
        try {
          tokenStream.reset();
          CharTermAttribute charTermAttribute = tokenStream.addAttribute(CharTermAttribute.class);
          while (tokenStream.incrementToken()) {
            token = charTermAttribute.toString();
            if (stopWordList != null && stopWordList.contains(token)) {
              continue;
            }
            docContent.increment(token);
          }
        } finally {
          tokenStream.close();
        }
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

      try {
        Path[] inputFiles = DistributedCache.getLocalCacheFiles(conf);
        if (inputFiles != null) {
          for (Path path : inputFiles) {
            // if (path.getName().startsWith(ParseCorpus.TERM)) {
            // stopWordList = ParseCorpus.importStopWordList(new BufferedReader(
            // new InputStreamReader(FileSystem.getLocal(conf).open(path), "utf-8")),
            // stopWordList);
            // } else {
            stopWordList = ParseCorpus.importStopWordList(new BufferedReader(new InputStreamReader(
                FileSystem.getLocal(conf).open(path), "utf-8")), stopWordList);
            // }
          }
        }
      } catch (IOException ioe) {
        ioe.printStackTrace();
      }

      Class<? extends Analyzer> analyzerClass = (Class<? extends Analyzer>) conf.getClass(
          Settings.PROPERTY_PREFIX + "parse.corpus.analyzer", null, Closeable.class);

      if (analyzerClass != null) {
        try {
          // sLogger.info("analyzerClass.getCanonicalName(): " + analyzerClass.getCanonicalName());
          // sLogger.info("analyzerClass.getName(): " + analyzerClass.getName());
          // sLogger.info("analyzerClass.getDeclaringClass(): " +
          // analyzerClass.getDeclaringClass());
          // sLogger.info("analyzerClass.getSuperClass(): " + analyzerClass.getSuperclass());
          // sLogger.info("analyzerClass.getSimpleName(): " + analyzerClass.getSimpleName());

          Constructor<?> cons = analyzerClass.getDeclaredConstructor(new Class[] { Version.class });
          // Constructor<?> cons = analyzerClass.getDeclaredConstructor(Version.class);
          // TODO: for some reason, bespin cluster does not support Lucene 4.0.0 at this point ---
          // always get java.lang.NoSuchFieldError: LUCENE_40, but it works in local.
          analyzer = (Analyzer) cons.newInstance(Version.LUCENE_35);

          // String[] examplesChinese = { "大家 晚上 好 ，我 的 名字 叫 Ke Zhai 。",
          // "日本 人民 要 牢牢 记住 ： “ 钓鱼岛 是 中国 神圣 不可 分割 的 领土 。 ” （ 续 ）",
          // "中国 进出口 银行 最近 在 日本 取得 债券 信用 等级 aa - 。" };

          // for (String text : examplesChinese) {
          // sLogger.info("Analyzing \"" + text + "\"");
          // String name = analyzer.getClass().getSimpleName();
          // sLogger.info("\t" + name + ":");
          // sLogger.info("\t");
          // TokenStream stream = analyzer.tokenStream("contents,",
          // new StringReader(new String(text.getBytes("UTF8"))));
          // stream.reset();
          // CharTermAttribute charTermAttribute = stream.addAttribute(CharTermAttribute.class);
          // while (stream.incrementToken()) {
          // sLogger.info("[" + charTermAttribute.toString() + "] ");
          // }
          // sLogger.info("\n");
          // }
        } catch (SecurityException e) {
          sLogger.error(e.getMessage());
        } catch (NoSuchMethodException e) {
          sLogger.error(e.getMessage());
        } catch (IllegalArgumentException e) {
          sLogger.error(e.getMessage());
        } catch (InstantiationException e) {
          sLogger.error(e.getMessage());
        } catch (IllegalAccessException e) {
          sLogger.error(e.getMessage());
        } catch (InvocationTargetException e) {
          sLogger.error(e.getMessage());
        }
      }
    }

    public void close() throws IOException {
      // analyzer.close();
      multipleOutputs.close();
    }
  }

  private static class TokenizeCombiner extends MapReduceBase implements
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

  private static class TokenizeReducer extends MapReduceBase implements
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

  public int[] tokenizeDocument(Configuration configuration, String inputPath, String outputPath,
      String stopwordPath, Class<? extends Analyzer> analyzerClass, int numberOfMappers,
      int numberOfReducers) throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName() + " - tokenize document");
    sLogger.info(" - input path: " + inputPath);
    sLogger.info(" - output path: " + outputPath);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + numberOfReducers);
    sLogger.info(" - analyzer class: "
        + (analyzerClass == null ? null : analyzerClass.getCanonicalName()));
    // sLogger.info(" - vocabulary path: " + vocabularyPath);
    sLogger.info(" - stopword list path: " + stopwordPath);

    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    conf.setJobName(ParseCorpus.class.getSimpleName() + " - tokenize document");

    MultipleOutputs.addMultiNamedOutput(conf, DOCUMENT, SequenceFileOutputFormat.class, Text.class,
        HMapSIW.class);
    MultipleOutputs.addMultiNamedOutput(conf, TITLE, SequenceFileOutputFormat.class, Text.class,
        NullWritable.class);

    if (analyzerClass != null) {
      conf.setClass(Settings.PROPERTY_PREFIX + "parse.corpus.analyzer", analyzerClass,
          Closeable.class);
      // conf.set(Settings.PROPERTY_PREFIX + "parse.corpus.analyzer", analyzerClass);
    }
    if (stopwordPath != null) {
      DistributedCache.addCacheFile(new Path(stopwordPath).toUri(), conf);
    }
    // if (vocabularyPath != null) {
    // DistributedCache.addCacheFile(new Path(vocabularyPath).toUri(), conf);
    // }

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
    int[] corpusStatistics = new int[2];

    corpusStatistics[0] = (int) counters.findCounter(MyCounter.TOTAL_DOCS).getCounter();
    sLogger.info("Total number of documents is: " + corpusStatistics[0]);

    corpusStatistics[1] = (int) counters.findCounter(MyCounter.TOTAL_TERMS).getCounter();
    sLogger.info("Total number of terms is: " + corpusStatistics[1]);

    return corpusStatistics;
  }

  public Path indexTitle(Configuration configuration, String inputTitles, String outputTitle,
      int numberOfMappers) throws Exception {
    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    FileSystem fs = FileSystem.get(conf);

    Path titleIndexPath = new Path(outputTitle);

    String outputTitleFile = titleIndexPath.getParent() + Path.SEPARATOR + Settings.TEMP
        + FileMerger.generateRandomString();

    // TODO: add in configuration for file merger object
    // FileMerger fm = new FileMerger();
    // fm.setConf(config);
    // Path titlePath = fm.mergeSequenceFiles(inputTitles, outputTitleFile, numberOfMappers,
    // Text.class, NullWritable.class, true);
    Path titlePath = FileMerger.mergeSequenceFiles(configuration, inputTitles, outputTitleFile,
        numberOfMappers, Text.class, NullWritable.class, true);

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

  private static class IndexTermMapper extends MapReduceBase implements
      Mapper<Text, PairOfInts, PairOfInts, Text> {
    float minimumDocumentCount = 0;
    float maximumDocumentCount = Float.MAX_VALUE;

    @SuppressWarnings("deprecation")
    public void map(Text key, PairOfInts value, OutputCollector<PairOfInts, Text> output,
        Reporter reporter) throws IOException {
      if (value.getLeftElement() < minimumDocumentCount) {
        reporter.incrCounter(MyCounter.LOW_DOCUMENT_FREQUENCY_TERMS, 1);
        return;
      }
      if (value.getLeftElement() > maximumDocumentCount) {
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

  private static class IndexTermReducer extends MapReduceBase implements
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

  public Path indexTerm(Configuration configuration, String inputTerms, String outputTerm,
      int numberOfMappers, float minimumDocumentCount, float maximumDocumentCount) throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName() + " - index term");
    sLogger.info(" - input path: " + inputTerms);
    sLogger.info(" - output path: " + outputTerm);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + 1);
    sLogger.info(" - minimum document count: " + minimumDocumentCount);
    sLogger.info(" - maximum document count: " + maximumDocumentCount);

    Path inputTermFiles = new Path(inputTerms);
    Path outputTermFile = new Path(outputTerm);

    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    FileSystem fs = FileSystem.get(conf);

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

    String outputString = outputTermFile.getParent() + Path.SEPARATOR + Settings.TEMP
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

  private static class IndexDocumentMapper extends MapReduceBase implements
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
          "How embarrassing! Could not find title " + key.toString() + " in index...");
      content.clear();
      itr = value.keySet().iterator();
      while (itr.hasNext()) {
        temp = itr.next();
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
              sLogger.info("Checking file in distributed cache: " + path.getName());
              sequenceFileReader = new SequenceFile.Reader(FileSystem.getLocal(conf), path, conf);

              if (path.getName().startsWith(TERM)) {
                Preconditions.checkArgument(termIndex == null,
                    "Term index was initialized already...");
                termIndex = ParseCorpus.importParameter(sequenceFileReader);
                // sLogger.info("Term index parameter imported as: " + path);
              } else if (path.getName().startsWith(TITLE)) {
                Preconditions.checkArgument(titleIndex == null,
                    "Title index was initialized already...");
                titleIndex = ParseCorpus.importParameter(sequenceFileReader);
                // sLogger.info("Title index parameter imported as: " + path);
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

  public Path indexDocument(Configuration configuration, String inputDocument,
      String outputDocument, String termIndex, String titleIndex, int numberOfMappers)
      throws Exception {
    sLogger.info("Tool: " + ParseCorpus.class.getSimpleName() + " - index document");
    sLogger.info(" - input path: " + inputDocument);
    sLogger.info(" - output path: " + outputDocument);
    sLogger.info(" - term index path: " + termIndex);
    sLogger.info(" - title index path: " + titleIndex);
    sLogger.info(" - number of mappers: " + numberOfMappers);
    sLogger.info(" - number of reducers: " + 0);

    Path inputDocumentFiles = new Path(inputDocument);
    Path outputDocumentFiles = new Path(outputDocument);
    Path termIndexPath = new Path(termIndex);
    Path titleIndexPath = new Path(titleIndex);

    JobConf conf = new JobConf(configuration, ParseCorpus.class);
    FileSystem fs = FileSystem.get(conf);

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
      if (intWritable.get() % 100000 == 0) {
        sLogger.info("Imported term " + text.toString() + " with index " + intWritable.toString());
      }
      hashMap.put(text.toString(), intWritable.get());
    }

    return hashMap;
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new ParseCorpus(), args);
    System.exit(res);
  }

  public static Set<String> importStopWordList(BufferedReader bufferedReader,
      Set<String> stopWordList) throws IOException {
    if (stopWordList == null) {
      stopWordList = new HashSet<String>();
    }

    String temp = bufferedReader.readLine();
    while (temp != null) {
      stopWordList.add(temp.trim());
      temp = bufferedReader.readLine();
    }

    return stopWordList;
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