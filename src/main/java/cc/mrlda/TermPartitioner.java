package cc.mrlda;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.Partitioner;

import edu.umd.cloud9.io.pair.PairOfInts;

public class TermPartitioner implements Partitioner<PairOfInts, DoubleWritable> {
  public int getPartition(PairOfInts key, DoubleWritable value, int numReduceTasks) {
    return (key.getLeftElement() & Integer.MAX_VALUE) % numReduceTasks;
  }

  public void configure(JobConf conf) {
  }
}