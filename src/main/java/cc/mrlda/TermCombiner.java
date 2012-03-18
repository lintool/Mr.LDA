package cc.mrlda;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.math.LogMath;

public class TermCombiner extends MapReduceBase implements
    Reducer<PairOfInts, DoubleWritable, PairOfInts, DoubleWritable> {
  private DoubleWritable outputValue = new DoubleWritable();

  public void reduce(PairOfInts key, Iterator<DoubleWritable> values,
      OutputCollector<PairOfInts, DoubleWritable> output, Reporter reporter) throws IOException {
    double sum = values.next().get();
    if (key.getLeftElement() <= 0) {
      // this is not a phi value
      while (values.hasNext()) {
        sum += values.next().get();
      }
    } else {
      // this is a phi value
      while (values.hasNext()) {
        sum = LogMath.add(sum, values.next().get());
      }
    }
    outputValue.set(sum);
    output.collect(key, outputValue);
  }
}