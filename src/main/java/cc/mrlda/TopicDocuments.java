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


import java.io.IOException;
import java.lang.Math;
import java.util.Map;
import java.util.Iterator;
import java.util.Set;
import java.util.HashSet;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
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
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.RunningJob;
import org.apache.hadoop.mapred.SequenceFileInputFormat;
import org.apache.hadoop.mapred.SequenceFileOutputFormat;
import org.apache.hadoop.mapred.lib.MultipleOutputs;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.MapII;

public class TopicDocuments extends Configured implements Tool {
	static final Logger sLogger = Logger.getLogger(TopicDocuments.class);


	public static class TopicDocumentMapper extends MapReduceBase implements Mapper<IntWritable,Document,IntWritable,TitledDocument> {
		private double[] gammas = null;
		private boolean multiply = false;
		
		public void map(IntWritable docTitle, Document doc, OutputCollector<IntWritable, TitledDocument> output, Reporter reporter){
			try {
				if(docTitle == null || doc == null){ return; }	
				double[] gammas = doc.getGamma();
				if(gammas == null){
					return;
				}
				if(multiply){
					int[] idxs = TopicUtils.sortedIndices(gammas.clone());
					doc.setGamma(null);
					TitledDocument td = new TitledDocument(docTitle,doc);
					double sum3 = 0;
					int numTop = 3;
					int multiplier = 5;
					for(int i = 0; i<numTop; i++){
						sum3 += gammas[idxs[i]];
					}
					sum3/=multiplier;
					for(int i = 0; i<numTop; i++){
						IntWritable idx = new IntWritable(idxs[i]);
						for(int m = 0; m < Math.round(gammas[idxs[i]]/sum3); m++){
							output.collect(idx, td);		
						}
					}
				} else {
					int idx = TopicUtils.maxIndex(gammas);
					doc.setGamma(null);
					TitledDocument td = new TitledDocument(docTitle,doc);
					output.collect(new IntWritable(idx), td);
				}
			} catch(Exception e) {
				reporter.incrCounter("Errors", "MapperExceptions", 1);
				System.err.println("Encountered exception during mapper "+e.getMessage() + e.getStackTrace());
			}	
		}
		public void configure(JobConf conf){
			multiply = conf.getBoolean(Settings.PROPERTY_PREFIX+"multiply", false);	
		}
	}

	public static class TopicDocumentReducer extends MapReduceBase implements Reducer<IntWritable,TitledDocument,IntWritable,Document> {
		private MultipleOutputs multipleOutputs;

		@Override
		public void reduce(IntWritable maxIdx, Iterator<TitledDocument> topicDocuments,
				OutputCollector<IntWritable, Document> output, Reporter reporter)
						throws IOException {
			System.out.println("Writing output for topic "+maxIdx.toString());
			OutputCollector outputCollector = multipleOutputs.getCollector("topic", maxIdx.toString(), reporter); 
			int cnt=1;
			while(topicDocuments.hasNext()){
				TitledDocument topicDocument = topicDocuments.next();
				
				if(topicDocument == null){ continue; }
				try { outputCollector.collect(topicDocument.getTitle(), topicDocument.getDocument()); }
				catch (Exception e) {
					reporter.incrCounter("Errors", "ReducerExceptions", 1);
					System.err.println("Caught exception "+e.getLocalizedMessage()); 
					if(topicDocument != null && topicDocument.getTitle() != null){ System.err.println("Blame "+topicDocument.getTitle().toString());} 
				}
				if(++cnt%10000==0) { reporter.incrCounter("ReducerProgress", "Topic"+maxIdx.toString(), 10000); }
			}
			
		}

		public void configure(JobConf conf) {
			multipleOutputs = new MultipleOutputs(conf);
		}


		public void close() throws IOException {
			multipleOutputs.close();
		}

	}
	

	public void setupHadoopJob(String inputs, String output, boolean multiply) throws Exception{
	
		JobConf conf = new JobConf(getConf(), TopicDocuments.class);
		FileSystem fs = FileSystem.get(conf);
	
		sLogger.info("Tool: " + TopicDocuments.class.getSimpleName());
		sLogger.info(" - input path: " + inputs);
		sLogger.info(" - output path: " + output);
	
		conf.setJobName(TopicDocuments.class.getSimpleName() + " - printin' some documents");
	
		conf.setMapperClass(TopicDocumentMapper.class);
		conf.setReducerClass(TopicDocumentReducer.class);
	
		conf.setMapOutputKeyClass(IntWritable.class);
		conf.setMapOutputValueClass(TitledDocument.class);
		conf.setOutputKeyClass(IntWritable.class);
		conf.setOutputValueClass(Document.class);
	
		conf.setInputFormat(SequenceFileInputFormat.class);
		conf.setOutputFormat(SequenceFileOutputFormat.class);
	
		//conf.setInt(Settings.PROPERTY_PREFIX + "topics", topics);
		conf.setBoolean(Settings.PROPERTY_PREFIX+"multiply", multiply);
		
		Path inputPath = new Path(inputs);
		Path outputPath = new Path(output);    
		fs.delete(outputPath, true);
	
		FileInputFormat.setInputPaths(conf, inputPath);
		FileOutputFormat.setOutputPath(conf, outputPath);
		FileOutputFormat.setCompressOutput(conf, true);
		
	
		MultipleOutputs.addMultiNamedOutput(conf, "topic", SequenceFileOutputFormat.class, IntWritable.class, Document.class);
		
		long startTime = System.currentTimeMillis();
		RunningJob job = JobClient.runJob(conf);
		sLogger.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0
				+ " seconds");
		fs.delete(new Path(outputPath+"/part*"));
		
	}

	private void setupLocalJob(String gammaString, String indexString, int topic) throws Exception{
	
		JobConf conf = new JobConf(TopicDocuments.class);
		FileSystem fs = FileSystem.get(conf);
		Map<Integer,String> termIndex = null;
		if(indexString != null){
			Path indexPath = new Path(indexString);
			Preconditions.checkArgument(fs.exists(indexPath) && fs.isFile(indexPath),
					"Invalid index path...");
			termIndex = TopicUtils.initializeIndex(fs,conf,indexPath);
	
		}
		
		Path gammaPath = new Path(gammaString);
		//      Preconditions.checkArgument(fs.exists(gammaPath) && fs.isFile(gammaPath), "Invalid gamma path...");
		Preconditions.checkArgument(fs.exists(gammaPath), "Invalid gamma path...");
		
		if(fs.getFileStatus(gammaPath).isDir()){
			try {
				FileStatus[] stat = fs.listStatus(gammaPath);
				for (int i = 0; i < stat.length; ++i) {
					processDocumentFile(fs, conf, stat[i].getPath(), termIndex, topic);
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		} else {
			processDocumentFile(fs,conf,gammaPath,termIndex, topic);
		}
	}

	public void processDocumentFile(FileSystem fs, JobConf conf, Path gammaPath, Map<Integer,String> termIndex, int topic) throws Exception{
		SequenceFile.Reader sequenceFileReader = null;
		try {
			sequenceFileReader = new SequenceFile.Reader(fs, gammaPath, conf);
			IntWritable intWritable = new IntWritable();
			Document doc = new Document();
			while (sequenceFileReader.next(intWritable,doc)) {
				double[] gammas = doc.getGamma();
				if(gammas == null){
					continue;
				} 
				int maxTopic = TopicUtils.maxIndex(gammas);
				if(maxTopic == topic){
					printDocument(doc, termIndex);
				}
			}
		} finally {
			IOUtils.closeStream(sequenceFileReader);
		}
	}

	public void printDocument(Document doc, Map <Integer, String> termIdx) {
		HMapII content = doc.getContent();
		System.out.print(content.size());
		boolean useTermIdx = (termIdx != null);
		for(MapII.Entry e : content.entrySet()){
			String key = Integer.toString(e.getKey());
			if(useTermIdx && termIdx.containsKey(e.getKey())){
					key = termIdx.get(e.getKey());
			}
			System.out.print(" "+key+':'+e.getValue());				
		}
		System.out.print("\n");
	}


	@SuppressWarnings("unchecked")
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(Settings.HELP_OPTION, false, "print the help message");
		options.addOption("multiply", false, "Create multiple copies of record for top 3 topics");
		options.addOption("hadoop", false, "Run distributed, create outputs on Hadoop");
	   options.addOption("raw",false,"Do not use term index - output raw ids");
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("input gamma file").create(Settings.INPUT_OPTION));
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("output path").create(Settings.OUTPUT_OPTION));
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("term index file").create(ParseCorpus.INDEX));
		options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
				.withDescription("topic to extract").create("topic"));

		String gammaString = null;
		String outputString = null;
		String indexString = null;
		boolean multiply = false;
		boolean hadoop = false;
		boolean raw = false;
		int topic = 0;

		CommandLineParser parser = new GnuParser();
		HelpFormatter formatter = new HelpFormatter();
		try {
			CommandLine line = parser.parse(options, args);
			
			if (line.hasOption(Settings.HELP_OPTION)) {
				formatter.printHelp(TopicDocuments.class.getName(), options);
				System.exit(0);
			}
	      if(line.hasOption("raw")){
	      	raw = true;
	      }
			if (line.hasOption("multiply")) {
				multiply = true;
			}
			if (line.hasOption("hadoop")) {
				hadoop = true;
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
			if (line.hasOption("topic")) {
				topic = Integer.parseInt(line.getOptionValue("topic"));
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
			setupHadoopJob(gammaString, outputString, multiply);
		} else {
			setupLocalJob(gammaString,indexString, topic);

		}

		
		return 0;
	}
	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new TopicDocuments(), args);
		System.exit(res);
	}
}