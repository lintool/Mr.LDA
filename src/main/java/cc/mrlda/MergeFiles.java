package cc.mrlda;

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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import edu.umd.cloud9.io.FileMerger;

public class MergeFiles extends Configured implements Tool {

	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(Settings.HELP_OPTION, false, "print the help message");
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("input file(s) or directory").isRequired().create(Settings.INPUT_OPTION));
		options.addOption(OptionBuilder.withArgName(Settings.PATH_INDICATOR).hasArg()
				.withDescription("output directory").isRequired().create(Settings.OUTPUT_OPTION));
		

		String inputPath = null;
		String outputPath = null;

		Configuration config = getConf();
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

		} catch (ParseException pe) {
			System.err.println(pe.getMessage());
			formatter.printHelp(MergeFiles.class.getName(), options);
			System.exit(0);	
		} catch (NumberFormatException nfe) {
			System.err.println(nfe.getMessage());
			System.exit(0);
		}

	    FileMerger fm = new FileMerger();
	    fm.setConf(config);
	    fm.mergeSequenceFiles(inputPath, outputPath, 25, IntWritable.class, Document.class, false);
		
		return 0;
	}

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new Configuration(), new MergeFiles(), args);
		System.exit(res);
	}

}
