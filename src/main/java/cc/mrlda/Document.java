package cc.mrlda;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.Iterator;

import org.apache.hadoop.io.Writable;

import edu.umd.cloud9.util.map.HMapII;
import edu.umd.cloud9.util.map.MapII;

public class Document implements Writable, Cloneable, Serializable {
  /**
   * 
   */
  private static final long serialVersionUID = 752244298258266755L;

  /**
   * 
   */
  private HMapII content = null;

  /**
   * @deprecated
   */
  private double[] gamma = null;

  /**
   * Define the total number of words in this document, not necessarily distinct.
   */
  private int numberOfWords = 0;

  /**
   * Creates a <code>LDADocument</code> object from a byte array.
   * 
   * @param bytes raw serialized representation
   * @return a newly-created <code>LDADocument</code> object
   * @throws IOException
   */
  public static Document create(byte[] bytes) throws IOException {
    return create(new DataInputStream(new ByteArrayInputStream(bytes)));
  }

  /**
   * Creates a <code>LDADocument</code> object from a <code>DataInput</code>.
   * 
   * @param in source for reading the serialized representation
   * @return a newly-created <code>LDADocument</code> object
   * @throws IOException
   */
  public static Document create(DataInput in) throws IOException {
    Document m = new Document();
    m.readFields(in);

    return m;
  }

  public Document() {
  }

  public Document(HMapII document) {
    this.content = document;
    if (document != null) {
      Iterator<Integer> itr = this.content.values().iterator();
      while (itr.hasNext()) {
        numberOfWords += itr.next();
      }
    }
  }

  /**
   * @deprecated
   * @param document
   * @param gamma
   */
  public Document(HMapII document, double[] gamma) {
    this(document);
    this.gamma = gamma;
  }

  /**
   * @deprecated
   * @param document
   * @param numberOfTopics
   */
  public Document(HMapII document, int numberOfTopics) {
    this(document, new double[numberOfTopics]);
  }

  public HMapII getContent() {
    return this.content;
  }

  /**
   * @deprecated
   * @return
   */
  public double[] getGamma() {
    return gamma;
  }

  /**
   * @deprecated
   * @return
   */
  public int getNumberOfTopics() {
    if (gamma == null) {
      return 0;
    } else {
      return gamma.length;
    }
  }

  /**
   * Get the total number of distinct types in this document.
   * 
   * @return the total number of unique types in this document.
   */
  public int getNumberOfTypes() {
    if (content == null) {
      return 0;
    } else {
      return content.size();
    }
  }

  /**
   * Get the total number of words in this document, not necessarily distinct.
   * 
   * @return the total number of words in this document, not necessarily distinct.
   */
  public int getNumberOfWords() {
    return numberOfWords;
  }

  /**
   * Deserializes the LDADocument.
   * 
   * @param in source for raw byte representation
   */
  public void readFields(DataInput in) throws IOException {
    numberOfWords = 0;

    int numEntries = in.readInt();
    if (numEntries <= 0) {
      content = null;
    } else {
      content = new HMapII();
      for (int i = 0; i < numEntries; i++) {
        int id = in.readInt();
        int count = in.readInt();
        content.put(id, count);
        numberOfWords += count;
      }
    }

    int numTopics = in.readInt();
    if (numTopics <= 0) {
      gamma = null;
    } else {
      gamma = new double[numTopics];
      for (int i = 0; i < numTopics; i++) {
        gamma[i] = in.readDouble();
      }
    }
  }

  /**
   * Returns the serialized representation of this object as a byte array.
   * 
   * @return byte array representing the serialized representation of this object
   * @throws IOException
   */
  public byte[] serialize() throws IOException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    DataOutputStream dataOut = new DataOutputStream(bytesOut);
    write(dataOut);

    return bytesOut.toByteArray();
  }

  public void setDocument(HMapII document) {
    this.content = document;
    numberOfWords = 0;

    if (document != null) {
      Iterator<Integer> itr = this.content.values().iterator();
      while (itr.hasNext()) {
        numberOfWords += itr.next();
      }
    }
  }

  /**
   * @deprecated
   * @param gamma
   */
  public void setGamma(double[] gamma) {
    this.gamma = gamma;
  }

  @Override
  public String toString() {
    StringBuilder document = new StringBuilder("content:\t");
    if (content == null) {
      document.append("null");
    } else {
      Iterator<Integer> itr = this.content.keySet().iterator();
      while (itr.hasNext()) {
        int id = itr.next();
        document.append(id);
        document.append(":");
        document.append(content.get(id));
        document.append(" ");
      }
    }
    document.append("\ngamma:\t");
    if (gamma == null) {
      document.append("null");
    } else {
      for (double value : gamma) {
        document.append(value);
        document.append(" ");
      }
    }

    return document.toString();
  }

  /**
   * Serializes the map.
   * 
   * @param out where to write the raw byte representation
   */
  public void write(DataOutput out) throws IOException {
    // Write out the number of entries in the map.
    if (content == null) {
      out.writeInt(0);
    } else {
      out.writeInt(content.size());
      for (MapII.Entry e : content.entrySet()) {
        out.writeInt(e.getKey());
        out.writeInt(e.getValue());
      }
    }

    // Write out the gamma values for this document.
    if (gamma == null) {
      out.writeInt(0);
    } else {
      out.writeInt(gamma.length);
      for (double value : gamma) {
        // TODO: change it to double and also in read method
        out.writeDouble(value);
      }
    }
  }
}