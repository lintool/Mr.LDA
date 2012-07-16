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
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapIFW;
import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.MapII;
import edu.umd.cloud9.io.pair.PairOfIntFloat;

import cc.mrlda.Document;

public class AllTopicDocuments extends Configured implements Tool {
    
    public Map<Integer,String> initializeTermIndex(FileSystem fs, Path indexPath, JobConf conf) throws Exception{
	SequenceFile.Reader sequenceFileReader = null;
	Map<Integer, String> termIndex = new HashMap<Integer, String>();
	try {
	    IntWritable intWritable = new IntWritable();
	    Text text = new Text();
	    sequenceFileReader = new SequenceFile.Reader(fs, indexPath, conf);
	    while (sequenceFileReader.next(intWritable, text)) {
		termIndex.put(intWritable.get(), text.toString());
	    }
	} finally {
	    IOUtils.closeStream(sequenceFileReader);
	}
	return termIndex;
    }
    public void processDocumentFile(FileSystem fs, Path gammaPath, JobConf conf, Map<Integer,String> termIndex, int topic) throws Exception{
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
		int maxTopic = maxIndex(gammas);
		System.out.print(maxTopic+"\t");
		printDocument(doc, termIndex);

	    }
	} finally {
	    IOUtils.closeStream(sequenceFileReader);
	}
    }

    public void printDocument(Document doc, Map <Integer, String> termIdx) {
	HMapII content = doc.getContent();
	System.out.print(content.size());
	for(MapII.Entry e : content.entrySet()){
	    if(termIdx.containsKey(e.getKey())){
		    System.out.print(" "+termIdx.get(e.getKey())+':'+e.getValue());
	    }
	}
	System.out.print("\n");
    }
		      

    public int maxIndex(double[] vals){
	int maxIdx = -1;
	for(int i = 0; i < vals.length; i++){
	    if(maxIdx < 0 || vals[maxIdx] < vals[i]){
		maxIdx = i;
	    }
	}
	return maxIdx;
    }



  @SuppressWarnings("unchecked")
      public int run(String[] args) throws Exception {
      Options options = new Options();
      
      options.addOption(Settings.HELP_OPTION, false, "print the help message");
      options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
			.withDescription("input gamma file").create(Settings.INPUT_OPTION));
      options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
			.withDescription("term index file").create(ParseCorpus.INDEX));
      options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
			.withDescription("topic to extract").create("topic"));
			
      String gammaString = null;
      String indexString = null;
      int topic = 0;

      CommandLineParser parser = new GnuParser();
      HelpFormatter formatter = new HelpFormatter();
      try {
	  CommandLine line = parser.parse(options, args);

	  if (line.hasOption(Settings.HELP_OPTION)) {
	      formatter.printHelp(AllTopicDocuments.class.getName(), options);
	      System.exit(0);
	  }

	  if (line.hasOption(Settings.INPUT_OPTION)) {
	      gammaString = line.getOptionValue(Settings.INPUT_OPTION);
	  } else {
	      throw new ParseException("Parsing failed due to " + Settings.INPUT_OPTION
				       + " not initialized...");
	  }

	  if (line.hasOption(ParseCorpus.INDEX)) {
	      indexString = line.getOptionValue(ParseCorpus.INDEX);
	  } else {
	      throw new ParseException("Parsing failed due to " + ParseCorpus.INDEX
				       + " not initialized...");
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
		      
      JobConf conf = new JobConf(AllTopicDocuments.class);
      FileSystem fs = FileSystem.get(conf);

      Path indexPath = new Path(indexString);
      Preconditions.checkArgument(fs.exists(indexPath) && fs.isFile(indexPath),
				  "Invalid index path...");
      Map<Integer,String> termIndex = initializeTermIndex(fs,indexPath,conf);

      Path gammaPath = new Path(gammaString);
      //      Preconditions.checkArgument(fs.exists(gammaPath) && fs.isFile(gammaPath), "Invalid gamma path...");
      Preconditions.checkArgument(fs.exists(gammaPath), "Invalid gamma path...");
      if(fs.getFileStatus(gammaPath).isDir()){
	  try {
	      FileStatus[] stat = fs.listStatus(gammaPath);
	      for (int i = 0; i < stat.length; ++i) {
		  processDocumentFile(fs, stat[i].getPath(), conf, termIndex, topic);
	      }
	  } catch (IOException e) {
	      e.printStackTrace();
	  }
      } else {
	  processDocumentFile(fs,gammaPath,conf,termIndex, topic);
      }
      
      return 0;
  }
    public static void main(String[] args) throws Exception {
	int res = ToolRunner.run(new Configuration(), new AllTopicDocuments(), args);
	System.exit(res);
    }
}