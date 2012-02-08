/*
 * Cloud9: A MapReduce Library for Hadoop
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You may
 * obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package cc.mrlda;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.Iterator;

import junit.framework.JUnit4TestAdapter;

import org.junit.Test;

import edu.umd.cloud9.io.map.HMapIFW;
import edu.umd.cloud9.io.map.HMapIIW;
import edu.umd.cloud9.io.map.HMapIIWTest;
import edu.umd.cloud9.util.map.HMapII;

public class LDADocumentTest {
  public static float PRECISION = 1e-6f;

  @Test
  public void testSerialize1() throws IOException {
    HMapII hmap1 = new HMapII();
    hmap1.put(1, 22);
    hmap1.put(2, 5);
    hmap1.put(3, 10);

    float[] array1 = new float[2];
    array1[0] = 0.238573f;
    array1[1] = 1.59382f;

    LDADocument doc1 = new LDADocument(hmap1, array1);
    assertEquals(doc1.getNumberOfTopics(), 2);
    assertEquals(doc1.getNumberOfWords(), 37);
    assertEquals(doc1.getNumberOfTypes(), 3);

    LDADocument doc2 = LDADocument.create(doc1.serialize());
    HMapII hmap2 = doc2.getContent();
    float[] array2 = doc2.getGamma();

    assertEquals(doc2.getNumberOfWords(), doc1.getNumberOfWords());
    assertEquals(doc2.getNumberOfTypes(), doc1.getNumberOfTypes());
    assertEquals(doc2.getNumberOfTopics(), doc1.getNumberOfTopics());
    assertEquals(hmap2.size(), hmap1.size());
    assertEquals(array2.length, array1.length);

    Iterator<Integer> itr = hmap2.keySet().iterator();
    while (itr.hasNext()) {
      int key = itr.next();
      assertEquals(hmap2.get(key), hmap1.get(key));
    }

    for (int i = 0; i < array2.length; i++) {
      assertEquals(array2[i], array1[i], PRECISION);
    }
  }

  @Test
  public void testSerialize2() throws IOException {
    HMapII hmap1 = new HMapII();
    hmap1.put(1, 22);
    hmap1.put(2, 5);
    hmap1.put(3, 10);
    float[] array1 = null;

    LDADocument doc1 = new LDADocument(hmap1, array1);

    assertEquals(doc1.getNumberOfTopics(), 0);
    assertEquals(doc1.getNumberOfWords(), 37);
    assertEquals(doc1.getNumberOfTypes(), 3);
    assertEquals(doc1.getGamma(), null);

    LDADocument doc2 = LDADocument.create(doc1.serialize());
    HMapII hmap2 = doc2.getContent();
    float[] array2 = doc2.getGamma();

    assertEquals(doc2.getNumberOfWords(), doc1.getNumberOfWords());
    assertEquals(doc2.getNumberOfTypes(), doc1.getNumberOfTypes());
    assertEquals(doc2.getNumberOfTopics(), doc1.getNumberOfTopics());
    assertEquals(array2, array1);
    assertEquals(hmap2.size(), hmap1.size());

    Iterator<Integer> itr = hmap2.keySet().iterator();
    while (itr.hasNext()) {
      int key = itr.next();
      assertEquals(hmap2.get(key), hmap1.get(key));
    }
  }

  @Test
  public void testSerialize3() throws IOException {
    HMapII hmap1 = null;
    float[] array1 = new float[2];
    array1[0] = 0.238573f;
    array1[1] = 1.59382f;

    LDADocument doc1 = new LDADocument(hmap1, array1);
    assertEquals(doc1.getNumberOfTopics(), 2);
    assertEquals(doc1.getNumberOfWords(), 0);
    assertEquals(doc1.getNumberOfTypes(), 0);
    assertEquals(doc1.getContent(), null);

    LDADocument doc2 = LDADocument.create(doc1.serialize());

    HMapII hmap2 = doc2.getContent();
    float[] array2 = doc2.getGamma();

    assertEquals(doc2.getNumberOfWords(), doc1.getNumberOfWords());
    assertEquals(doc2.getNumberOfTypes(), doc1.getNumberOfTypes());
    assertEquals(doc2.getNumberOfTopics(), doc1.getNumberOfTopics());
    assertEquals(hmap2, hmap1);
    assertEquals(array2.length, array1.length);

    for (int i = 0; i < array2.length; i++) {
      assertEquals(array2[i], array1[i], PRECISION);
    }
  }

  /**
   * Test
   * 
   * @throws IOException
   */
  @Test
  public void testSerialize4() throws IOException {
    HMapII hmap1 = null;
    float[] array1 = null;

    LDADocument doc1 = new LDADocument(hmap1, array1);
    assertEquals(doc1.getNumberOfTopics(), 0);
    assertEquals(doc1.getNumberOfWords(), 0);
    assertEquals(doc1.getNumberOfTypes(), 0);
    assertEquals(doc1.getContent(), null);
    assertEquals(doc1.getGamma(), null);

    LDADocument doc2 = LDADocument.create(doc1.serialize());

    HMapII hmap2 = doc2.getContent();
    float[] array2 = doc2.getGamma();

    assertEquals(doc2.getNumberOfWords(), doc1.getNumberOfWords());
    assertEquals(doc2.getNumberOfTypes(), doc1.getNumberOfTypes());
    assertEquals(doc2.getNumberOfTopics(), doc1.getNumberOfTopics());
    assertEquals(hmap2, hmap1);
    assertEquals(array2, array1);
  }

  /**
   * Test assigning new gamma
   * 
   * @throws IOException
   */
  @Test
  public void testSerialize5() throws IOException {
    HMapII hmap1 = new HMapII();
    hmap1.put(1, 22);
    hmap1.put(2, 5);
    hmap1.put(3, 10);

    LDADocument doc1 = new LDADocument(hmap1);
    assertEquals(doc1.getNumberOfWords(), 37);
    assertEquals(doc1.getNumberOfTypes(), 3);
    assertEquals(doc1.getNumberOfTopics(), 0);
    assertEquals(doc1.getGamma(), null);

    float[] array1 = new float[2];
    array1[0] = 0.238573f;
    array1[1] = 1.59382f;

    doc1.setGamma(array1);
    for (int i = 0; i < doc1.getGamma().length; i++) {
      assertEquals(doc1.getGamma()[i], array1[i], PRECISION);
    }

    LDADocument doc2 = LDADocument.create(doc1.serialize());
    HMapII hmap2 = doc2.getContent();
    float[] array2 = doc2.getGamma();

    assertEquals(doc2.getNumberOfWords(), doc1.getNumberOfWords());
    assertEquals(doc2.getNumberOfTypes(), doc1.getNumberOfTypes());
    assertEquals(doc2.getNumberOfTopics(), doc1.getNumberOfTopics());
    assertEquals(hmap2.size(), hmap1.size());
    assertEquals(array2.length, array1.length);

    Iterator<Integer> itr = hmap2.keySet().iterator();
    while (itr.hasNext()) {
      int key = itr.next();
      assertEquals(hmap2.get(key), hmap1.get(key));
    }

    for (int i = 0; i < array2.length; i++) {
      assertEquals(array2[i], array1[i], PRECISION);
    }
  }

  public static junit.framework.Test suite() {
    return new JUnit4TestAdapter(LDADocumentTest.class);
  }
}