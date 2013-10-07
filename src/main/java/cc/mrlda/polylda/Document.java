package cc.mrlda.polylda;

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
  private HMapII[] content = null;

  /**
   * @deprecated
   */
  private double[] gamma = null;

  /**
   * Define the total number of words in this document for every language, not necessarily distinct.
   */
  private int numberOfWords[] = null;

  /**
   * Define the total number of words in this document aggregated over all languages.
   */
  private int totalNumberOfWords = 0;

  public Document() {
  }

  public Document(HMapII[] document, double[] gamma) {
    this(document);
    this.gamma = gamma;
  }

  public Document(HMapII[] document, int numberOfTopics) {
    this(document, new double[numberOfTopics]);
  }

  public Document(HMapII[] document) {
    this.content = document;
    if (document != null) {
      this.numberOfWords = new int[document.length];

      for (int i = 0; i < this.content.length; i++) {
        if (this.content[i] != null) {
          Iterator<Integer> itr = this.content[i].values().iterator();
          while (itr.hasNext()) {
            numberOfWords[i] += itr.next();
          }
        }
        totalNumberOfWords += numberOfWords[i];
      }
    }
  }

  /**
   * Get the total number of distinct types in this document for this language.
   * 
   * @return the total number of unique types in this document for this language.
   */
  public int getNumberOfTypes(int languageIndex) throws ArrayIndexOutOfBoundsException {
    if (content == null || content[languageIndex] == null) {
      return 0;
    } else {
      return content[languageIndex].size();
    }
  }

  public int getNumberOfLanguages() {
    if (content == null) {
      return 0;
    }
    return content.length;
  }

  public int[] getNumberOfTypes() {
    if (content == null) {
      return null;
    }

    int[] numberOfTypes = new int[content.length];
    for (int i = 0; i < content.length; i++) {
      if (content[i] == null) {
        numberOfTypes[i] = 0;
      } else {
        numberOfTypes[i] = content[i].size();
      }
    }

    return numberOfTypes;
  }

  /**
   * Get the total number of words in this document, not necessarily distinct, for all languages.
   * 
   * @return the total number of words in this document, not necessarily distinct, for all
   *         languages.
   */
  public int[] getNumberOfWords() {
    return numberOfWords;
  }

  public int getTotalNumberOfWords() {
    return totalNumberOfWords;
  }

  public int getNumberOfWords(int languageIndex) throws ArrayIndexOutOfBoundsException {
    if (numberOfWords == null) {
      return 0;
    }
    return numberOfWords[languageIndex];
  }

  public HMapII[] getContent() {
    return this.content;
  }

  public HMapII getContent(int languageIndex) {
    if (this.content == null) {
      return null;
    }
    return this.content[languageIndex];
  }

  /**
   * @deprecated
   */
  public void resetGamma() {
    this.gamma = null;
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
   * @deprecated
   * @return
   */
  public double[] getGamma() {
    return gamma;
  }

  /**
   * @deprecated
   */
  public void setGamma(double[] gamma) {
    this.gamma = gamma;
  }

  /**
   * Deserializes the LDADocument.
   * 
   * @param in source for raw byte representation
   */
  public void readFields(DataInput in) throws IOException {
    int numLanguages = in.readInt();
    if (numLanguages <= 0) {
      content = null;
      numberOfWords = null;
      totalNumberOfWords = 0;
    } else {
      numberOfWords = new int[numLanguages];
      content = new HMapII[numLanguages];
      totalNumberOfWords = 0;

      for (int i = 0; i < numLanguages; i++) {
        int numEntries = in.readInt();

        if (numEntries <= 0) {
          content[i] = null;
        } else {
          content[i] = new HMapII();
          for (int j = 0; j < numEntries; j++) {
            int id = in.readInt();
            int count = in.readInt();
            content[i].put(id, count);
            numberOfWords[i] += count;
            totalNumberOfWords += count;
          }
        }
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

  public void setDocument(HMapII[] document) {
    this.content = document;
    if (this.content == null) {
      this.numberOfWords = null;
      this.totalNumberOfWords = 0;
      return;
    }

    this.numberOfWords = new int[content.length];
    this.totalNumberOfWords = 0;
    Iterator<Integer> itr = null;
    for (int i = 0; i < numberOfWords.length; i++) {
      if (this.content[i] == null) {
        numberOfWords[i] = 0;
      } else {
        itr = this.content[i].values().iterator();
        while (itr.hasNext()) {
          numberOfWords[i] += itr.next();
        }
      }
      this.totalNumberOfWords += numberOfWords[i];
    }
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
      out.writeInt(content.length);
      for (HMapII hmapii : content) {
        if (hmapii == null) {
          out.writeInt(0);
        } else {
          out.writeInt(hmapii.size());
          for (MapII.Entry e : hmapii.entrySet()) {
            out.writeInt(e.getKey());
            out.writeInt(e.getValue());
          }
        }
      }
    }

    // Write out the gamma values for this document.
    if (gamma == null) {
      out.writeInt(0);
    } else {
      out.writeInt(gamma.length);
      for (double value : gamma) {
        out.writeDouble(value);
      }
    }
  }

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

  @Override
  public String toString() {
    StringBuilder document = new StringBuilder("content:\n");
    if (content == null) {
      document.append("null");
    } else {
      for (int i = 0; i < content.length; i++) {
        document.append("language " + i + "\t");

        if (this.content[i] == null) {
          document.append("null\n");
        } else {
          Iterator<Integer> itr = this.content[i].keySet().iterator();
          while (itr.hasNext()) {
            int id = itr.next();
            document.append(id);
            document.append(":");
            document.append(content[i].get(id));
            document.append(" ");
          }
          document.append("\n");
        }
      }
    }

    document.append("gamma:\t");
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
}