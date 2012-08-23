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
import org.apache.hadoop.util.GenericOptionsParser;

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

public class ParseCorpusRaw extends Configured implements Tool {
	static final Logger sLogger = Logger.getLogger(ParseCorpusRaw.class);

	private static enum MyCounter {
		TOTAL_DOCS, TOTAL_TERMS,
	}

	public static final String DOCUMENT = "document";
	public static final String TERM = "term";
	public static final String TITLE = "title";
	public static final String INDEX = "index";

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


		String inputPath = null;
		String outputPath = null;
		int numberOfMappers = Settings.DEFAULT_NUMBER_OF_MAPPERS;
		int numberOfReducers = Settings.DEFAULT_NUMBER_OF_REDUCERS;
		// boolean localMerge = FileMerger.LOCAL_MERGE;


		Configuration config = getConf();
		CommandLineParser parser = new GnuParser();
		HelpFormatter formatter = new HelpFormatter();
		try {
			CommandLine line = parser.parse(options, args);

			if (line.hasOption(Settings.HELP_OPTION)) {
				formatter.printHelp(ParseCorpusRaw.class.getName(), options);
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
		} catch (ParseException pe) {
			System.err.println(pe.getMessage());
			formatter.printHelp(ParseCorpusRaw.class.getName(), options);
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
		FileSystem fs = FileSystem.get(new JobConf(config, ParseCorpusRaw.class));
		fs.delete(new Path(outputPath), true);

		try {
			tokenizeDocument(config, inputPath, outputPath + DOCUMENT, numberOfMappers, numberOfReducers);
		} catch (Exception e) {
			System.err.println(e.getMessage());
			System.exit(0);
		}

		return 0;
	}

	public static class TokenizeMapper extends MapReduceBase implements
	Mapper<LongWritable, Text, IntWritable, Document> {
		private Text term = new Text();
		private PairOfInts counts = new PairOfInts();

		private IntWritable docTitle = null;
		private HMapII docContent = null;
		private Iterator<String> itr = null;
		private String temp = null;

		@SuppressWarnings("deprecation")
		public void map(LongWritable key, Text value, OutputCollector<IntWritable, Document> output,
				Reporter reporter) throws IOException {

			temp = value.toString();
			int index = temp.indexOf(Settings.TAB);

			docTitle = new IntWritable(Integer.parseInt(temp.substring(0, index).trim()));
			docContent = new HMapII();
			String[] terms = temp.substring(index + 1).split("\\s+");
			for( String term : terms){
				int termInt = Integer.parseInt(term);
				docContent.increment(termInt);
			}

			output.collect(docTitle, new Document(docContent));

			reporter.incrCounter(MyCounter.TOTAL_DOCS, 1);
		}

	}

	public void tokenizeDocument(Configuration config, String inputPath, String outputPath, int numberOfMappers,
			int numberOfReducers ) throws Exception {
		sLogger.info("Tool: " + ParseCorpusRaw.class.getSimpleName());
		sLogger.info(" - input path: " + inputPath);
		sLogger.info(" - output path: " + outputPath);
		sLogger.info(" - number of mappers: " + numberOfMappers);
		sLogger.info(" - number of reducers: " + numberOfReducers);

		JobConf conf = new JobConf(config, ParseCorpusRaw.class);
		conf.setJobName(ParseCorpusRaw.class.getSimpleName() + " - tokenize document");
		FileSystem fs = FileSystem.get(conf);

		conf.setNumMapTasks(numberOfMappers);
		conf.setNumReduceTasks(0);

		conf.setMapperClass(TokenizeMapper.class);

		conf.setMapOutputKeyClass(IntWritable.class);
		conf.setMapOutputValueClass(Document.class);
		conf.setOutputKeyClass(IntWritable.class);
		conf.setOutputValueClass(Document.class);

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

		return;
	}



	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new ParseCorpusRaw(), args);
		System.exit(res);
	}

	/**
	 * @deprecated
	 * @param sequenceFileReader
	 * @param sequenceFileWriter
	 * @return
	 * @throws IOException
	 */
}