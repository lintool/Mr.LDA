package cc.mrlda;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;

public class TopicUtils {
	public static int[] sortedIndices(double[] vals){
		int[] indices = new int[vals.length];
		Set<Integer> availIndices = new HashSet<Integer>(vals.length);
		for(int i = 0; i < vals.length; i++){
			availIndices.add(new Integer(i));
		}
		int key = 0;
		while(!availIndices.isEmpty()){
			Integer maxIdx = new Integer(-1);
			Iterator<Integer> j = availIndices.iterator();
			while(j.hasNext()){
				Integer idx = j.next();
				if(maxIdx.intValue() < 0 || vals[idx.intValue()] > vals[maxIdx.intValue()]){
					maxIdx = idx;
				}
			}
			indices[key]=maxIdx.intValue();
			key++;
			availIndices.remove(maxIdx);
		}
		return indices;
	}

	public static int maxIndex(double[] vals){
		int maxIdx = -1;
		for(int i = 0; i < vals.length; i++){
			if(maxIdx < 0 || vals[maxIdx] < vals[i]){
				maxIdx = i;
			}
		}
		return maxIdx;
	}
	public static Map<Integer,String> initializeIndex(FileSystem fs, JobConf conf, Path indexPath) throws Exception{
		SequenceFile.Reader sequenceFileReader = null;
		Map<Integer, String> index = new HashMap<Integer, String>();
		try {
			IntWritable intWritable = new IntWritable();
			Text text = new Text();
			sequenceFileReader = new SequenceFile.Reader(fs, indexPath, conf);
			while (sequenceFileReader.next(intWritable, text)) {
				index.put(intWritable.get(),text.toString());
			}
		} catch (Exception e){
			System.err.println("Error loading index "+e.getMessage());
		} finally {
			IOUtils.closeStream(sequenceFileReader);
		}
		return index;
	}
	
	
}
