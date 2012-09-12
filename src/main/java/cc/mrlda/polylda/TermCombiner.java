package cc.mrlda.polylda;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.triple.TripleOfInts;
import edu.umd.cloud9.math.LogMath;

public class TermCombiner extends MapReduceBase implements
    Reducer<TripleOfInts, DoubleWritable, TripleOfInts, DoubleWritable> {
  private DoubleWritable outputValue = new DoubleWritable();

  public void reduce(TripleOfInts key, Iterator<DoubleWritable> values,
      OutputCollector<TripleOfInts, DoubleWritable> output, Reporter reporter) throws IOException {
    Preconditions.checkArgument(key.getLeftElement() >= 0, "Unexpected key pattern...");
    double sum = values.next().get();
    if (key.getLeftElement() == 0) {
      Preconditions.checkArgument(key.getMiddleElement() > 0 && key.getRightElement() == 0,
          "Unexpected key pattern...");
      // this is a alpha sufficient statistics term
      while (values.hasNext()) {
        sum += values.next().get();
      }
    } else {
      Preconditions.checkArgument(key.getMiddleElement() > 0 && key.getRightElement() > 0,
          "Unexpected key pattern...");
      // this is a phi value
      while (values.hasNext()) {
        sum = LogMath.add(sum, values.next().get());
      }
    }
    outputValue.set(sum);
    output.collect(key, outputValue);
  }
}