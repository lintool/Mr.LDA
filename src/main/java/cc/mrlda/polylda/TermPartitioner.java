package cc.mrlda.polylda;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import edu.umd.cloud9.io.triple.TripleOfInts;

public class TermPartitioner implements Partitioner<TripleOfInts, DoubleWritable> {
  public int getPartition(TripleOfInts key, DoubleWritable value, int numReduceTasks) {
    return (key.getLeftElement() * key.getMiddleElement() & Integer.MAX_VALUE) % numReduceTasks;
  }

  public void configure(JobConf conf) {
  }
}