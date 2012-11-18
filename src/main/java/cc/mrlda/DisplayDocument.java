package cc.mrlda;

import java.io.IOException;
import java.util.Iterator;

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
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import cc.mrlda.ParseCorpus.MyCounter;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.pair.PairOfIntFloat;
import edu.umd.cloud9.io.pair.PairOfInts;

public class DisplayDocument extends Configured implements Tool {
  @SuppressWarnings("unchecked")
  public int run(String[] args) throws Exception {
    Options options = new Options();

    options.addOption(Settings.HELP_OPTION, false, "print the help message");
    options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
        .withDescription("input document directory").create(Settings.INPUT_OPTION));

    String gammaString = null;

    CommandLineParser parser = new GnuParser();
    HelpFormatter formatter = new HelpFormatter();
    try {
      CommandLine line = parser.parse(options, args);

      if (line.hasOption(Settings.HELP_OPTION)) {
        formatter.printHelp(ParseCorpus.class.getName(), options);
        System.exit(0);
      }

      if (line.hasOption(Settings.INPUT_OPTION)) {
        gammaString = line.getOptionValue(Settings.INPUT_OPTION);
      } else {
        throw new ParseException("Parsing failed due to " + Settings.INPUT_OPTION
            + " not initialized...");
      }
    } catch (ParseException pe) {
      System.err.println(pe.getMessage());
      formatter.printHelp(ParseCorpus.class.getName(), options);
      System.exit(0);
    } catch (NumberFormatException nfe) {
      System.err.println(nfe.getMessage());
      System.exit(0);
    }

    JobConf conf = new JobConf(DisplayDocument.class);
    FileSystem fs = FileSystem.get(conf);

    Path gammaPath = new Path(gammaString);
    Preconditions.checkArgument(fs.exists(gammaPath) && !fs.isFile(gammaPath),
        "Invalid gamma path...");

    SequenceFile.Reader sequenceFileReader = null;
    try {
      IntWritable intWritable = new IntWritable();
      Document document = new Document();
      StringBuffer strBuf = new StringBuffer();

      for (FileStatus fileStatus : fs.listStatus(gammaPath)) {
        sequenceFileReader = new SequenceFile.Reader(fs, fileStatus.getPath(), conf);
        while (sequenceFileReader.next(intWritable, document)) {
          Preconditions.checkArgument(document.getGamma() != null
              && document.getGamma().length == document.getNumberOfTopics(),
              "topic distribution not specified for document " + intWritable.get());

          System.out.print(intWritable.get() + " ");
          for (int i = 0; i < document.getNumberOfTopics(); i++) {
            System.out.print(document.getGamma()[i] + " ");
          }
          System.out.print("\n");
        }
      }
    } finally {
      IOUtils.closeStream(sequenceFileReader);
    }

    return 0;
  }

  /**
   * @deprecated
   * @author kzhai
   */
  private static class DocumentMapper extends MapReduceBase implements
      Mapper<IntWritable, Document, PairOfIntFloat, IntWritable> {
    private PairOfIntFloat pairOfIntFloat = new PairOfIntFloat();
    private IntWritable intWritable = new IntWritable();

    @SuppressWarnings("deprecation")
    public void map(IntWritable key, Document value,
        OutputCollector<PairOfIntFloat, IntWritable> output, Reporter reporter) throws IOException {
      Preconditions.checkArgument(
          value.getGamma() != null && value.getGamma().length == value.getNumberOfTopics(),
          "topic distribution not specified for document " + key.get());
      for (int i = 0; i < value.getNumberOfTopics(); i++) {
        pairOfIntFloat.set(key.get(), (float) value.getGamma()[i]);
        intWritable.set(i);
        output.collect(pairOfIntFloat, intWritable);
      }
    }
  }

  public static void main(String[] args) throws Exception {
    int res = ToolRunner.run(new Configuration(), new DisplayDocument(), args);
    System.exit(res);
  }
}