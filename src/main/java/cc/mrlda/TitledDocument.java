package cc.mrlda;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.Serializable;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.WritableComparable;

public class TitledDocument implements WritableComparable<TitledDocument>,Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8676182049727111372L;
	private IntWritable docTitle;
	private Document docObj;
	
	public TitledDocument() {
		
	}
	
	public TitledDocument(IntWritable title, Document document) {
		System.out.println("Setting title to"+title.toString());
		docTitle = title;
		docObj = document;
	}
	
	public IntWritable getTitle() {
		return docTitle;
	}
	
	public Document getDocument() {
		return docObj;
	}
	
	@Override
	public void readFields(DataInput in) throws IOException {
		docTitle = new IntWritable();
		docTitle.readFields(in);
		docObj = new Document();
		docObj.readFields(in);
	}

	@Override
	public void write(DataOutput out) throws IOException {
		docTitle.write(out);
		docObj.write(out);
	}
	
	@Override
	public int compareTo(TitledDocument otherTD) {
		return docTitle.compareTo(otherTD.getTitle());
	}
}
