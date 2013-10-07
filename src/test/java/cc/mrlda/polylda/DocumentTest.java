package cc.mrlda.polylda;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Iterator;

import junit.framework.JUnit4TestAdapter;

import org.junit.Test;

import edu.umd.cloud9.util.map.HMapII;

public class DocumentTest {
  public static double PRECISION = 1e-12;

  @Test
  public void testConstructor1() {
    Document doc = new Document();
    assertTrue(doc.getGamma() == null);
    assertEquals(doc.getNumberOfTopics(), 0);
    assertEquals(doc.getTotalNumberOfWords(), 0);

    assertTrue(doc.getContent() == null);
    assertEquals(doc.getNumberOfTypes(), null);
    assertEquals(doc.getNumberOfWords(), null);
  }

  @Test
  public void testConstructor2() {
    HMapII[] hmaps = new HMapII[2];
    hmaps[0] = new HMapII();
    hmaps[0].put(6, 22);
    hmaps[0].put(12, 5);
    hmaps[0].put(23, 10);

    hmaps[1] = new HMapII();
    hmaps[1].put(1, 4);
    hmaps[1].put(2, 7);
    hmaps[1].put(3, 6);
    hmaps[1].put(4, 2);

    double[] array1 = new double[2];
    array1[0] = 0.238573f;
    array1[1] = 1.59382f;

    Document doc = new Document(hmaps, array1);

    assertTrue(doc.getGamma() != null);
    assertEquals(doc.getNumberOfTopics(), array1.length);

    assertTrue(doc.getContent() != null);
    assertEquals(doc.getNumberOfLanguages(), 2);
    assertEquals(doc.getNumberOfWords(0), 37);
    assertEquals(doc.getNumberOfTypes(0), hmaps[0].size());
    assertEquals(doc.getNumberOfWords(1), 19);
    assertEquals(doc.getNumberOfTypes(1), hmaps[1].size());
    assertEquals(doc.getTotalNumberOfWords(), 56);

    Iterator<Integer> itr = doc.getContent()[0].keySet().iterator();
    while (itr.hasNext()) {
      int key = itr.next();
      assertEquals(doc.getContent()[0].get(key), hmaps[0].get(key));
    }

    itr = doc.getContent()[1].keySet().iterator();
    while (itr.hasNext()) {
      int key = itr.next();
      assertEquals(doc.getContent()[1].get(key), hmaps[1].get(key));
    }
  }

  @Test
  public void testSetDocument() {
    Document doc = new Document();
    assertTrue(doc.getGamma() == null);
    assertEquals(doc.getNumberOfTopics(), 0);

    assertTrue(doc.getContent() == null);
    assertEquals(doc.getNumberOfTypes(), null);
    assertEquals(doc.getNumberOfWords(), null);
    assertEquals(doc.getTotalNumberOfWords(), 0);

    assertEquals(doc.getNumberOfLanguages(), 0);
    assertEquals(doc.getNumberOfTypes(0), 0);
    assertEquals(doc.getNumberOfWords(0), 0);

    HMapII[] hmap = new HMapII[3];
    hmap[1] = new HMapII();
    hmap[1].put(1, 22);
    hmap[1].put(2, 5);
    hmap[1].put(3, 10);

    doc.setDocument(hmap);
    assertTrue(doc.getGamma() == null);
    assertEquals(doc.getNumberOfTopics(), 0);

    assertTrue(doc.getContent() != null);
    assertEquals(doc.getNumberOfLanguages(), 3);
    assertEquals(doc.getNumberOfTypes(0), 0);
    assertEquals(doc.getNumberOfTypes(1), hmap[1].size());
    assertEquals(doc.getNumberOfTypes(2), 0);
    assertEquals(doc.getNumberOfWords(0), 0);
    assertEquals(doc.getNumberOfWords(1), 37);
    assertEquals(doc.getNumberOfWords(2), 0);
    assertEquals(doc.getTotalNumberOfWords(), 37);

    for (int i = 0; i < doc.getNumberOfLanguages(); i++) {
      if (doc.getContent()[i] != null) {
        Iterator<Integer> itr = doc.getContent()[i].keySet().iterator();
        while (itr.hasNext()) {
          int key = itr.next();
          assertEquals(doc.getContent()[i].get(key), hmap[i].get(key));
        }
      } else {
        assertEquals(doc.getContent()[i], hmap[i]);
      }
    }

    doc.setDocument(null);
    assertTrue(doc.getGamma() == null);
    assertEquals(doc.getNumberOfTopics(), 0);

    assertTrue(doc.getContent() == null);
    assertEquals(doc.getNumberOfTypes(), null);
    assertEquals(doc.getNumberOfWords(), null);
    assertEquals(doc.getTotalNumberOfWords(), 0);

    assertEquals(doc.getNumberOfLanguages(), 0);
    assertEquals(doc.getNumberOfTypes(0), 0);
    assertEquals(doc.getNumberOfWords(0), 0);
  }

  @Test
  public void testSerialize1() throws IOException {
    HMapII[] hmap1 = new HMapII[3];
    hmap1[1] = new HMapII();
    hmap1[1].put(1, 22);
    hmap1[1].put(2, 5);
    hmap1[1].put(3, 10);

    double[] array1 = new double[2];
    array1[0] = 0.238573f;
    array1[1] = 1.59382f;

    Document doc1 = new Document(hmap1, array1);

    Document doc2 = Document.create(doc1.serialize());
    HMapII[] hmap2 = doc2.getContent();
    double[] array2 = doc2.getGamma();

    assertTrue(doc2.getGamma() != null);
    assertEquals(doc2.getNumberOfTopics(), 2);

    for (int i = 0; i < array2.length; i++) {
      assertEquals(array2[i], array1[i], PRECISION);
    }

    assertTrue(hmap2 != null);
    assertEquals(doc2.getNumberOfLanguages(), 3);
    assertEquals(doc2.getNumberOfTypes(0), 0);
    assertEquals(doc2.getNumberOfTypes(1), hmap1[1].size());
    assertEquals(doc2.getNumberOfTypes(2), 0);
    assertEquals(doc2.getNumberOfWords(0), 0);
    assertEquals(doc2.getNumberOfWords(1), 37);
    assertEquals(doc2.getNumberOfWords(2), 0);
    assertEquals(doc2.getTotalNumberOfWords(), 37);

    for (int i = 0; i < doc2.getNumberOfLanguages(); i++) {
      if (doc2.getContent()[i] != null) {
        Iterator<Integer> itr = hmap2[i].keySet().iterator();
        while (itr.hasNext()) {
          int key = itr.next();
          assertEquals(hmap2[i].get(key), hmap1[i].get(key));
        }
      } else {
        assertEquals(hmap2[i], hmap1[i]);
      }
    }

    doc2.setDocument(null);
    doc2.setGamma(null);
    doc1 = Document.create(doc2.serialize());

    assertTrue(doc1.getGamma() == null);
    assertEquals(doc1.getNumberOfTopics(), 0);

    assertEquals(doc1.getNumberOfLanguages(), 0);
    assertEquals(doc1.getContent(), null);
    assertEquals(doc1.getContent(0), null);
    assertEquals(doc1.getNumberOfTypes(0), 0);
    assertEquals(doc1.getNumberOfWords(), null);
    assertEquals(doc1.getNumberOfWords(0), 0);
    assertEquals(doc1.getTotalNumberOfWords(), 0);
  }

  @Test
  public void testSerialize2() throws IOException {
    HMapII[] hmap1 = new HMapII[3];
    hmap1[1] = new HMapII();
    hmap1[1].put(1, 22);
    hmap1[1].put(2, 5);
    hmap1[1].put(3, 10);
    double[] array1 = null;

    Document doc1 = new Document(hmap1, array1);

    assertEquals(doc1.getGamma(), null);
    assertEquals(doc1.getNumberOfTopics(), 0);

    Document doc2 = Document.create(doc1.serialize());
    HMapII[] hmap2 = doc2.getContent();
    double[] array2 = doc2.getGamma();

    assertEquals(array2, array1);
    assertEquals(doc2.getNumberOfTopics(), doc1.getNumberOfTopics());

    assertEquals(doc2.getNumberOfLanguages(), doc1.getNumberOfLanguages());
    assertEquals(doc2.getTotalNumberOfWords(), doc1.getTotalNumberOfWords());

    for (int i = 0; i < doc2.getNumberOfLanguages(); i++) {
      assertEquals(doc2.getNumberOfWords(i), doc1.getNumberOfWords(i));
      assertEquals(doc2.getNumberOfTypes(i), doc1.getNumberOfTypes(i));

      if (doc2.getContent()[i] != null) {
        Iterator<Integer> itr = hmap2[i].keySet().iterator();
        while (itr.hasNext()) {
          int key = itr.next();
          assertEquals(hmap2[i].get(key), hmap1[i].get(key));
        }
      } else {
        assertEquals(hmap2[i], hmap1[i]);
      }
    }
  }

  @Test
  public void testSerialize3() throws IOException {
    HMapII[] hmap1 = null;
    double[] array1 = new double[2];
    array1[0] = 0.238573f;
    array1[1] = 1.59382f;

    Document doc1 = new Document(hmap1, array1);
    assertEquals(doc1.getNumberOfTopics(), 2);
    assertEquals(doc1.getNumberOfWords(), null);
    assertEquals(doc1.getNumberOfTypes(), null);
    assertEquals(doc1.getContent(), null);

    Document doc2 = Document.create(doc1.serialize());

    HMapII[] hmap2 = doc2.getContent();
    double[] array2 = doc2.getGamma();

    assertEquals(doc2.getNumberOfWords(), doc1.getNumberOfWords());
    assertEquals(doc2.getNumberOfTypes(), doc1.getNumberOfTypes());
    assertEquals(doc2.getNumberOfTopics(), doc1.getNumberOfTopics());
    assertEquals(doc2.getNumberOfLanguages(), doc1.getNumberOfLanguages());
    assertEquals(doc2.getTotalNumberOfWords(), doc1.getTotalNumberOfWords());

    assertEquals(doc2.getContent(), null);
    assertEquals(doc2.getContent(0), null);
    assertEquals(array2.length, array1.length);

    for (int i = 0; i < array2.length; i++) {
      assertEquals(array2[i], array1[i], PRECISION);
    }
  }

  @Test
  public void testSerialize4() throws IOException {
    HMapII[] hmap1 = null;
    double[] array1 = null;

    Document doc1 = new Document(hmap1, array1);
    assertEquals(doc1.getNumberOfLanguages(), 0);
    assertEquals(doc1.getNumberOfTopics(), 0);
    assertEquals(doc1.getNumberOfWords(), null);
    assertEquals(doc1.getTotalNumberOfWords(), 0);

    assertEquals(doc1.getNumberOfTypes(), null);
    assertEquals(doc1.getContent(), null);
    assertEquals(doc1.getGamma(), null);

    Document doc2 = Document.create(doc1.serialize());

    HMapII[] hmap2 = doc2.getContent();
    double[] array2 = doc2.getGamma();

    assertEquals(doc2.getNumberOfLanguages(), 0);
    assertEquals(doc2.getNumberOfTopics(), 0);
    assertEquals(doc2.getNumberOfWords(), null);
    assertEquals(doc2.getTotalNumberOfWords(), 0);

    assertEquals(doc2.getNumberOfTypes(), null);
    assertEquals(doc2.getContent(), null);
    assertEquals(doc2.getGamma(), null);
  }

  public static junit.framework.Test suite() {
    return new JUnit4TestAdapter(DocumentTest.class);
  }
}