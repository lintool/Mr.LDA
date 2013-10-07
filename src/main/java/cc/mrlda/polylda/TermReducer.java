package cc.mrlda.polylda;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapIDW;
import edu.umd.cloud9.io.pair.PairOfIntFloat;
import edu.umd.cloud9.io.triple.TripleOfInts;
import edu.umd.cloud9.math.Gamma;
import edu.umd.cloud9.math.LogMath;

public class TermReducer extends MapReduceBase implements
    Reducer<TripleOfInts, DoubleWritable, IntWritable, DoubleWritable> {
  // private TripleOfIntsDouble outputKey = new TripleOfIntsDouble();
  private PairOfIntFloat outputKey = new PairOfIntFloat();
  private HMapIDW outputValue = new HMapIDW();

  private static boolean learning = Settings.LEARNING_MODE;
  private MultipleOutputs multipleOutputs;

  private IntWritable intWritable = new IntWritable();
  private DoubleWritable doubleWritable = new DoubleWritable();

  private int topicIndex = 0;
  private int languageIndex = 0;
  private double normalizeFactor = 0;

  private OutputCollector<PairOfIntFloat, HMapIDW> outputBeta;

  public void configure(JobConf conf) {
    multipleOutputs = new MultipleOutputs(conf);

    learning = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.train", Settings.LEARNING_MODE);

    // System.out.println("======================================================================");
    // System.out.println("Available processors (cores): " +
    // Runtime.getRuntime().availableProcessors());
    // long maxMemory = Runtime.getRuntime().maxMemory();
    // System.out.println("Maximum memory (bytes): " + (maxMemory == Long.MAX_VALUE ? "no limit" :
    // maxMemory));
    // System.out.println("Free memory (bytes): " + Runtime.getRuntime().freeMemory());
    // System.out.println("Total memory (bytes): " + Runtime.getRuntime().totalMemory());
    // System.out.println("======================================================================");
  }

  public void reduce(TripleOfInts key, Iterator<DoubleWritable> values,
      OutputCollector<IntWritable, DoubleWritable> output, Reporter reporter) throws IOException {
    Preconditions.checkArgument(learning, "Invalid key from Mapper...");

    // if this value is the sufficient statistics for alpha updating
    if (key.getLeftElement() == 0) {
      double sum = values.next().get();
      while (values.hasNext()) {
        sum += values.next().get();
      }

      Preconditions.checkArgument(key.getMiddleElement() > 0 && key.getRightElement() == 0,
          "Unexpected sequence order for alpha sufficient statistics: " + key.toString());

      intWritable.set(key.getMiddleElement());
      doubleWritable.set(sum);
      output.collect(intWritable, doubleWritable);

      return;
    }

    double logPhiValue = values.next().get();
    while (values.hasNext()) {
      logPhiValue = LogMath.add(logPhiValue, values.next().get());
    }

    // get the beta output for this language, language index starts from 1
    if (languageIndex != key.getLeftElement()) {
      languageIndex = key.getLeftElement();
      outputBeta = multipleOutputs.getCollector(Settings.BETA, Settings.LANGUAGE_INDICATOR
          + languageIndex, reporter);
    }

    // TODO: correct this count
    reporter.incrCounter(ParseCorpus.TOTAL_TERMS_IN_LANGUAGE, Settings.LANGUAGE_OPTION
        + languageIndex, 1);

    // topic index starts from 1
    if (topicIndex != key.getMiddleElement()) {
      if (topicIndex != 0) {
        outputKey.set(topicIndex, (float) Gamma.digamma(Math.exp(normalizeFactor)));
        outputBeta.collect(outputKey, outputValue);
      }

      topicIndex = key.getMiddleElement();
      normalizeFactor = logPhiValue;

      outputValue.clear();
      outputValue.put(key.getRightElement(), Gamma.digamma(Math.exp(logPhiValue)));
    } else {
      normalizeFactor = LogMath.add(normalizeFactor, logPhiValue);
      outputValue.put(key.getRightElement(), Gamma.digamma(Math.exp(logPhiValue)));
    }
  }

  public void close() throws IOException {
    if (!outputValue.isEmpty()) {
      outputKey.set(topicIndex, (float) Gamma.digamma(Math.exp(normalizeFactor)));
      outputBeta.collect(outputKey, outputValue);
    }

    multipleOutputs.close();
  }
}