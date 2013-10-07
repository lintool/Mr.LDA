package cc.mrlda;

import java.io.IOException;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.lib.MultipleOutputs;

import cc.mrlda.VariationalInference.ParameterCounter;

import com.google.common.base.Preconditions;

import edu.umd.cloud9.io.map.HMapIDW;
import edu.umd.cloud9.io.pair.PairOfIntFloat;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.math.Gamma;
import edu.umd.cloud9.math.LogMath;
import edu.umd.cloud9.util.map.HMapIV;

public class TermReducer extends MapReduceBase implements
    Reducer<PairOfInts, DoubleWritable, IntWritable, DoubleWritable> {
  // boolean approximateBeta = false;
  // boolean truncateBeta = false;
  // int truncationSize = 10000;
  // TreeMap<Double, Integer> treeMap = new TreeMap<Double, Integer>();
  // Iterator<Entry<Double, Integer>> itr = null;

  private static HMapIV<Set<Integer>> lambdaMap = null;

  private static boolean learning = Settings.LEARNING_MODE;
  // private static int numberOfTerms = 0;

  private int topicIndex = 0;
  private double logNormalizeFactor = 0;

  private MultipleOutputs multipleOutputs;
  private OutputCollector<PairOfIntFloat, HMapIDW> outputBeta;
  // private OutputCollector<PairOfIntFloat, ProbDist> outputBeta;
  // private OutputCollector<PairOfIntFloat, HashMap> outputBeta;
  // private OutputCollector<PairOfIntFloat, BloomMap> outputBeta;

  private IntWritable intWritable = new IntWritable();
  private DoubleWritable doubleWritable = new DoubleWritable();

  private PairOfIntFloat outputKey = new PairOfIntFloat();

  // private HashMap outputValue = null;
  // private BloomMap outputValue = null;
  // private ProbDist outputValue = null;
  private HMapIDW outputValue = null;

  public void configure(JobConf conf) {
    multipleOutputs = new MultipleOutputs(conf);

    learning = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.train", Settings.LEARNING_MODE);

    // truncateBeta = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.truncate.beta", false);

    // outputValue = new HashMap();
    outputValue = new HMapIDW();

    // approximateBeta = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.truncate.beta", false);
    // if (!approximateBeta) {
    // outputValue = new HashMap();
    // } else {
    // outputValue = new BloomMap(numberOfTerms * 100, 3, Hash.JENKINS_HASH);
    // }

    boolean informedPrior = conf.getBoolean(Settings.PROPERTY_PREFIX + "model.informed.prior",
        false);

    Path[] inputFiles;
    SequenceFile.Reader sequenceFileReader = null;

    try {
      inputFiles = DistributedCache.getLocalCacheFiles(conf);

      if (inputFiles != null) {
        for (Path path : inputFiles) {
          try {
            sequenceFileReader = new SequenceFile.Reader(FileSystem.getLocal(conf), path, conf);

            if (path.getName().startsWith(Settings.ALPHA)) {
              continue;
            } else if (path.getName().startsWith(Settings.BETA)) {
              continue;
            } else if (path.getName().startsWith(InformedPrior.ETA)) {
              Preconditions.checkArgument(lambdaMap == null,
                  "Lambda matrix was initialized already...");
              lambdaMap = InformedPrior.importEta(sequenceFileReader);
            } else {
              throw new IllegalArgumentException("Unexpected file in distributed cache: "
                  + path.getName());
            }
          } catch (IllegalArgumentException iae) {
            iae.printStackTrace();
          } catch (IOException ioe) {
            ioe.printStackTrace();
          } finally {
            IOUtils.closeStream(sequenceFileReader);
          }
        }
      }
    } catch (IOException ioe) {
      ioe.printStackTrace();
    }

    Preconditions.checkArgument(informedPrior == (lambdaMap != null),
        "Fail to initialize informed prior...");

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

  public void reduce(PairOfInts key, Iterator<DoubleWritable> values,
      OutputCollector<IntWritable, DoubleWritable> output, Reporter reporter) throws IOException {
    // if this value is the sufficient statistics for alpha updating
    if (key.getLeftElement() == 0) {
      double sum = values.next().get();
      while (values.hasNext()) {
        sum += values.next().get();
      }

      Preconditions.checkArgument(key.getRightElement() > 0,
          "Unexpected sequence order for alpha sufficient statistics: " + key.toString());

      intWritable.set(key.getRightElement());
      doubleWritable.set(sum);
      output.collect(intWritable, doubleWritable);

      return;
    }

    // I would be very surprised to get here...
    Preconditions.checkArgument(learning, "Invalid key from Mapper");
    reporter.incrCounter(ParameterCounter.TOTAL_TERMS, 1);

    double logPhiValue = values.next().get();
    while (values.hasNext()) {
      logPhiValue = LogMath.add(logPhiValue, values.next().get());
    }

    if (lambdaMap != null) {
      logPhiValue = LogMath.add(
          InformedPrior.getLogEta(key.getRightElement(), lambdaMap.get(topicIndex)), logPhiValue);
    } else {
      logPhiValue = LogMath.add(Settings.DEFAULT_LOG_ETA, logPhiValue);
    }

    if (topicIndex != key.getLeftElement()) {
      if (topicIndex == 0) {
        outputBeta = multipleOutputs.getCollector(Settings.BETA, Settings.BETA, reporter);
      } else {
        outputKey.set(topicIndex, (float) Gamma.digamma(Math.exp(logNormalizeFactor)));

        // if (truncateBeta) {
        // itr = treeMap.entrySet().iterator();
        // Entry<Double, Integer> temp = null;
        // outputValue.clear();
        // while (itr.hasNext()) {
        // temp = itr.next();
        // outputValue.put(temp.getValue(), temp.getKey());
        // }
        // }

        outputBeta.collect(outputKey, outputValue);
      }

      topicIndex = key.getLeftElement();
      logNormalizeFactor = logPhiValue;
      // if (truncateBeta) {
      // treeMap.clear();
      // treeMap.put(phiValue, key.getRightElement());
      // } else {
      outputValue.clear();
      outputValue.put(key.getRightElement(), Gamma.digamma(Math.exp(logPhiValue)));
      // }
    } else {
      // if (truncateBeta) {
      // if (treeMap.size() >= truncationSize) {
      // if (treeMap.firstKey() < phiValue) {
      // normalizeFactor = Math.log(Math.exp(normalizeFactor) - Math.exp(treeMap.firstKey()));
      // treeMap.remove(treeMap.firstKey());
      //
      // treeMap.put(phiValue, key.getRightElement());
      // normalizeFactor = LogMath.add(normalizeFactor, phiValue);
      // }
      // } else {
      // treeMap.put(phiValue, key.getRightElement());
      // normalizeFactor = LogMath.add(normalizeFactor, phiValue);
      // }
      // } else {
      logNormalizeFactor = LogMath.add(logNormalizeFactor, logPhiValue);
      outputValue.put(key.getRightElement(), Gamma.digamma(Math.exp(logPhiValue)));
      // }
    }
  }

  public void close() throws IOException {
    // if (truncateBeta) {
    // if (!treeMap.isEmpty()) {
    // outputKey.set(topicIndex, (float) normalizeFactor);
    // itr = treeMap.entrySet().iterator();
    // Entry<Double, Integer> temp = null;
    // outputValue.clear();
    // while (itr.hasNext()) {
    // temp = itr.next();
    // outputValue.put(temp.getValue(), temp.getKey().floatValue());
    // }
    // outputBeta.collect(outputKey, outputValue);
    // }
    // } else {
    if (!outputValue.isEmpty()) {
      outputKey.set(topicIndex, (float) Gamma.digamma(Math.exp(logNormalizeFactor)));
      outputBeta.collect(outputKey, outputValue);
    }
    // }
    multipleOutputs.close();
  }
}