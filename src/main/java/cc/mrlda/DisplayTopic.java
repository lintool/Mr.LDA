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
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapIDW;
import edu.umd.cloud9.io.pair.PairOfIntFloat;

public class DisplayTopic extends Configured implements Tool {
  public static final String TOP_DISPLAY_OPTION = "topdisplay";
  public static final int TOP_DISPLAY = 10;

  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(Settings.HELP_OPTION, false, "print the help message");
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input beta file").create(Settings.INPUT_OPTION));
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("term index file").create(ParseCorpus.INDEX));
    options.addOption(OptionBuilder.withArgName(Settings.INTEGER_INDICATOR).hasArg()
        .withDescription("display top terms only (default - 10)").create(TOP_DISPLAY_OPTION));

    String betaString = null;
    String indexString = null;
    int topDisplay = TOP_DISPLAY;

    CommandLineParser parser = new GnuParser();
    HelpFormatter formatter = new HelpFormatter();
    try {
      CommandLine line = parser.parse(options, args);

      if (line.hasOption(Settings.HELP_OPTION)) {
        formatter.printHelp(ParseCorpus.class.getName(), options);
        System.exit(0);
      }

      if (line.hasOption(Settings.INPUT_OPTION)) {
        betaString = line.getOptionValue(Settings.INPUT_OPTION);
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

      if (line.hasOption(TOP_DISPLAY_OPTION)) {
        topDisplay = Integer.parseInt(line.getOptionValue(TOP_DISPLAY_OPTION));
      }
    } catch (ParseException pe) {
      System.err.println(pe.getMessage());
      formatter.printHelp(ParseCorpus.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      System.err.println(nfe.getMessage());
      System.exit(0);
    }

    JobConf conf = new JobConf(DisplayTopic.class);
    FileSystem fs = FileSystem.get(conf);

    Path indexPath = new Path(indexString);
    Preconditions.checkArgument(fs.exists(indexPath) && fs.isFile(indexPath),
        "Invalid index path...");

    Path betaPath = new Path(betaString);
    Preconditions.checkArgument(fs.exists(betaPath) && fs.isFile(betaPath), "Invalid beta path...");

    SequenceFile.Reader sequenceFileReader = null;
    try {
      IntWritable intWritable = new IntWritable();
      Text text = new Text();
      Map<Integer, String> termIndex = new HashMap<Integer, String>();
      sequenceFileReader = new SequenceFile.Reader(fs, indexPath, conf);
      while (sequenceFileReader.next(intWritable, text)) {
        termIndex.put(intWritable.get(), text.toString());
      }

      PairOfIntFloat pairOfIntFloat = new PairOfIntFloat();
      // HMapIFW hmap = new HMapIFW();
      HMapIDW hmap = new HMapIDW();
      TreeMap<Double, Integer> treeMap = new TreeMap<Double, Integer>();
      sequenceFileReader = new SequenceFile.Reader(fs, betaPath, conf);
      while (sequenceFileReader.next(pairOfIntFloat, hmap)) {
        treeMap.clear();

        System.out.println("==============================");
        System.out.println("Top ranked " + topDisplay + " terms for Topic "
            + pairOfIntFloat.getLeftElement());
        System.out.println("==============================");

        Iterator<Integer> itr1 = hmap.keySet().iterator();
        int temp1 = 0;
        while (itr1.hasNext()) {
          temp1 = itr1.next();
          treeMap.put(-hmap.get(temp1), temp1);
          if (treeMap.size() > topDisplay) {
            treeMap.remove(treeMap.lastKey());
          }
        }

        Iterator<Double> itr2 = treeMap.keySet().iterator();
        double temp2 = 0;
        while (itr2.hasNext()) {
          temp2 = itr2.next();
          if (termIndex.containsKey(treeMap.get(temp2))) {
            System.out.println(termIndex.get(treeMap.get(temp2)) + "\t\t" + -temp2);
          } else {
            System.out.println("How embarrassing! Term index not found...");
          }
        }
      }
    } finally {
      IOUtils.closeStream(sequenceFileReader);
    }

    return 0;
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new DisplayTopic(), args);
    System.exit(res);
  }
}