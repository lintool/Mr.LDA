package org.apache.hadoop.mapred;

import java.io.IOException;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.SequenceFile.CompressionType;
import org.apache.hadoop.io.compress.CompressionCodec;
import org.apache.hadoop.io.compress.DefaultCodec;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCommitter;
import org.apache.hadoop.mapred.RecordWriter;
import org.apache.hadoop.mapred.TaskAttemptContext;
import org.apache.hadoop.util.Progressable;
import org.apache.hadoop.util.ReflectionUtils;

/**
 * Extension to {@link SequenceFileOutputFormat} which avoids committing files that have no output in a *very* hacky way.
 * 
 * @author Ke Zhai
 * 
 * @param <K>
 * @param <V>
 */
public class NonEmptySequenceFileOutputFormat<K, V> extends SequenceFileOutputFormat<K, V> {
  /** Flag to track whether anything was output */
  protected boolean outputWritten = false;

  /**
   * Wrap the parent {@link RecordWriter} to allow us to track whether
   * {@link Context#write(Object, Object)} was actually called
   */
  @Override
  public RecordWriter<K, V> getRecordWriter(FileSystem ignored, JobConf job, String name,
      Progressable progress) throws IOException {
    // get the path of the temporary output file
    final Path file = FileOutputFormat.getTaskOutputPath(job, name);

    final FileSystem fs = file.getFileSystem(job);
    CompressionCodec codec = null;
    CompressionType compressionType = CompressionType.NONE;
    if (getCompressOutput(job)) {
      // find the kind of compression to do
      compressionType = getOutputCompressionType(job);

      // find the right codec
      Class<? extends CompressionCodec> codecClass = getOutputCompressorClass(job,
          DefaultCodec.class);
      codec = ReflectionUtils.newInstance(codecClass, job);
    }
    final SequenceFile.Writer out = SequenceFile.createWriter(fs, job, file,
        job.getOutputKeyClass(), job.getOutputValueClass(), compressionType, codec, progress);

    return new RecordWriter<K, V>() {

      public void write(K key, V value) throws IOException {
        outputWritten = true;
        out.append(key, value);
      }

      public void close(Reporter reporter) throws IOException {
        out.close();
        
        // this is to tell the system, if that output stream did write anything, destroy the file completely
        if (! outputWritten){
         fs.delete(file, true);
        }
      }
    };
  }
}