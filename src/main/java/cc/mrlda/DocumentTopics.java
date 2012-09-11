/*
 * Ivory: A Hadoop toolkit for Web-scale information retrieval
 * 
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You may
 * obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0 
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package cc.mrlda;

import java.lang.Integer;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.TreeMap;
import java.io.IOException;

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
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.TextOutputFormat;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapIFW;
import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.MapII;
import edu.umd.cloud9.io.pair.PairOfIntFloat;

import cc.mrlda.Document;
import cc.mrlda.TopicDocuments.TopicDocumentMapper;
import cc.mrlda.TopicDocuments.TopicDocumentReducer;

public class DocumentTopics extends Configured implements Tool {
	static final Logger sLogger = Logger.getLogger(TopicDocuments.class);
	
	public static String[] processDocument(Integer docKey, Document doc, Map<Integer,String> titleIndex){
		String outputKey;
		String outputValue;
		double[] gammas = doc.getGamma();
		if(gammas == null){
			return null;
		}
		String title = docKey.toString();
		if(titleIndex != null){
			if(!titleIndex.containsKey(docKey)){
				return null;
			}
			title = titleIndex.get(docKey);
		}
		outputKey = title;
		outputValue = "";
		for(int i = 0; i < gammas.length; i++){
			outputValue+= "\t"+i+":"+gammas[i];
		}
		String[] x = {outputKey,outputValue};
		return x;
	}
	
	public void processFile(FileSystem fs, JobConf conf, Path gammaPath,  Map<Integer,String> titleIndex) throws Exception{
		SequenceFile.Reader sequenceFileReader = null;
		try {
			sequenceFileReader = new SequenceFile.Reader(fs, gammaPath, conf);
			IntWritable intWritable = new IntWritable();
			Document doc = new Document();
			Integer docKey = null;
			while (sequenceFileReader.next(intWritable,doc)) {
				String[] output = processDocument(new Integer(intWritable.get()),doc,titleIndex);
				System.out.println(output[0]+"\t"+output[1]);
			}
		} finally {
			IOUtils.closeStream(sequenceFileReader);
		}
	}


	public static class DocumentTopicMapper extends MapReduceBase implements Mapper<IntWritable,Document,Text,Text> {
		private boolean useIndex = false;
		private Map<Integer,String> titleIndex = null;
		
		@Override
		public void map(IntWritable docKey, Document doc,
				OutputCollector<Text, Text> outputCollector, Reporter reporter)
				throws IOException {
			String[] output = DocumentTopics.processDocument(new Integer(docKey.get()),doc,titleIndex);
			outputCollector.collect(new Text(output[0]),new Text(output[1]));
		}
		
		public void configure(JobConf conf) {
			useIndex = conf.getBoolean(Settings.PROPERTY_PREFIX+"use_index", false);
			if(useIndex){
				try{
					String indexName = conf.get(Settings.PROPERTY_PREFIX+"index_name");
					Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);
					for(Path cachePath : cacheFiles){
						if(cachePath.getName().equals(indexName)){
							System.out.println("Loading "+cachePath.toString());
							titleIndex = TopicUtils.initializeIndex(FileSystem.getLocal(conf), conf, cachePath);
						}
					}
				} catch (IOException ioe) {
					ioe.printStackTrace();
				} catch (Exception e) {
					System.err.println(e.getMessage()); 
					e.printStackTrace();
				}
				
			}
		}


		public void close() throws IOException {
			
		}

	}
	
	public void setupHadoopJob(String inputs, String output, String indexString) throws Exception{

		JobConf conf = new JobConf(getConf(), DocumentTopics.class);
		FileSystem fs = FileSystem.get(conf);

		sLogger.info("Tool: " + DocumentTopics.class.getSimpleName());
		sLogger.info(" - input path: " + inputs);
		sLogger.info(" - output path: " + output);

		conf.setJobName(DocumentTopics.class.getSimpleName() + " - printin' some documents");

		if(indexString != null){
			Path indexPath = new Path(indexString);
			Preconditions.checkArgument(fs.exists(indexPath) && fs.isFile(indexPath), "Bad index path "+indexString+" - aborting");
			DistributedCache.addCacheFile(indexPath.toUri(), conf);
			conf.setBoolean(Settings.PROPERTY_PREFIX + "use_index", true);
			conf.set(Settings.PROPERTY_PREFIX + "index_name", indexPath.getName());
			sLogger.info(" - index path: " + indexString);
		}
		
		conf.setMapperClass(DocumentTopicMapper.class);
		//conf.setReducerClass(TopicDocumentReducer.class);

		conf.setMapOutputKeyClass(Text.class);
		conf.setMapOutputValueClass(Text.class);
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(Text.class);

		conf.setInputFormat(SequenceFileInputFormat.class);
		//conf.setOutputFormat(SequenceFileOutputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);
		
		Path inputPath = new Path(inputs);
		Path outputPath = new Path(output);    
		fs.delete(outputPath, true);

		FileInputFormat.setInputPaths(conf, inputPath);
		FileOutputFormat.setOutputPath(conf, outputPath);
//		FileOutputFormat.setCompressOutput(conf, true);
		

		
		long startTime = System.currentTimeMillis();
		RunningJob job = JobClient.runJob(conf);
		sLogger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0
				+ " seconds");
	}
	
	public void setupLocalJob(String gammaString, String indexString) throws Exception{

		JobConf conf = new JobConf(DocumentTopics.class);
		FileSystem fs = FileSystem.get(conf);
		
		Map<Integer,String> titleIndex = null;
		if(indexString != null){
			Path titleIndexPath = new Path(indexString);
			Preconditions.checkArgument(fs.exists(titleIndexPath) && fs.isFile(titleIndexPath),"Invalid index path...");
			titleIndex = TopicUtils.initializeIndex(fs, conf, titleIndexPath);
		}
		
		Path gammaPath = new Path(gammaString);
		//      Preconditions.checkArgument(fs.exists(gammaPath) && fs.isFile(gammaPath), "Invalid gamma path...");
		Preconditions.checkArgument(fs.exists(gammaPath), "Invalid gamma path...");
		if(fs.getFileStatus(gammaPath).isDir()){
			try {
				FileStatus[] stat = fs.listStatus(gammaPath);
				for (int i = 0; i < stat.length; ++i) {
					processFile(fs, conf, stat[i].getPath(), titleIndex);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		} else {
			processFile(fs,conf, gammaPath,titleIndex);
		}
	}
	
	

	@SuppressWarnings("unchecked")
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(Settings.HELP_OPTION, false, "print the help message");
		options.addOption("hadoop", false, "Run distributed, create outputs on Hadoop");
	    options.addOption("raw",false,"Do not use term index - output raw ids");
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("output path").create(Settings.OUTPUT_OPTION));
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("input gamma file").create(Settings.INPUT_OPTION));
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("title index path").create(ParseCorpus.INDEX));
		
		String gammaString = null;
		String indexString = null;
		String outputString = null;
		boolean hadoop = false;
		boolean raw = false;

		CommandLineParser parser = new GnuParser();
		HelpFormatter formatter = new HelpFormatter();
		try {
			CommandLine line = parser.parse(options, args);
			if (line.hasOption(Settings.HELP_OPTION)) {
				formatter.printHelp(DocumentTopics.class.getName(), options);
				System.exit(0);
			}
			if (line.hasOption("hadoop")) {
				hadoop = true;
			}
	      if(line.hasOption("raw")){
	      	raw = true;
	      }

			if (line.hasOption(Settings.INPUT_OPTION)) {
				gammaString = line.getOptionValue(Settings.INPUT_OPTION);
			} else {
				throw new ParseException("Parsing failed due to " + Settings.INPUT_OPTION
						+ " not initialized...");
			}
			if (line.hasOption(Settings.OUTPUT_OPTION)) {
				outputString = line.getOptionValue(Settings.OUTPUT_OPTION);
			} else {
				if(hadoop){
					throw new ParseException("Parsing failed due to " + Settings.OUTPUT_OPTION
							+ " not initialized...");
				}
			}
	      if(!raw){
	      	if (line.hasOption(ParseCorpus.INDEX)) {
	      		indexString = line.getOptionValue(ParseCorpus.INDEX);
	      	} else {
	      		throw new ParseException("Parsing failed due to " + ParseCorpus.INDEX
	      				+ " not initialized...");
	      	}
	      }

		} catch (ParseException pe) {
			System.err.println(pe.getMessage());
			formatter.printHelp(ParseCorpus.class.getName(), options);
			System.exit(0);
		} catch (NumberFormatException nfe) {
			System.err.println(nfe.getMessage());
			System.exit(0);
		}


		if(hadoop){
			setupHadoopJob(gammaString, outputString, indexString);
		} else {
			setupLocalJob(gammaString, indexString);
		}


		return 0;
	}
	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new DocumentTopics(), args);
		System.exit(res);
	}
}