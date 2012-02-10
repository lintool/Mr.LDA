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

package cc.mrlda.util;

import static org.junit.Assert.assertEquals;
import junit.framework.JUnit4TestAdapter;

import org.junit.Test;

import cern.jet.stat.Gamma;

public class ApproximationTest {
  public static double PRECISION_6 = 1e-6f;
  public static double PRECISION_9 = 1e-9f;
  public static double PRECISION_12 = 1e-12f;

  @Test
  public void testDigamma() {
    assertEquals(Approximation.digamma(1000000), 13.81551005796419, PRECISION_12);
    assertEquals(Approximation.digamma(100000), 11.512920464961896, PRECISION_12);
    assertEquals(Approximation.digamma(10000), 9.21029037114285, PRECISION_12);

    assertEquals(Approximation.digamma(1000), 6.907255195648812, PRECISION_12);
    assertEquals(Approximation.digamma(100), 4.600161852738087, PRECISION_12);
    assertEquals(Approximation.digamma(10), 2.2517525890667214, PRECISION_12);

    assertEquals(Approximation.digamma(1), -0.5772156649015328, PRECISION_12);
    assertEquals(Approximation.digamma(0.1), -10.42375494041107, PRECISION_12);
    assertEquals(Approximation.digamma(0.01), -100.56088545786886, PRECISION_12);

    // precision drops down accordingly when computing digamma function for small value
    assertEquals(Approximation.digamma(0.001), -1000.5755719318336, PRECISION_9);
    assertEquals(Approximation.digamma(0.0001), -10000.57705117741, PRECISION_6);
    assertEquals(Approximation.digamma(0.00001), -100000.57719922789, PRECISION_6);

    assertEquals(Approximation.digamma(-0.001), 999.4211381980015, PRECISION_9);

    assertEquals(Approximation.digamma(-0.01), 99.4062136959443, PRECISION_12);
    assertEquals(Approximation.digamma(-0.1), 9.245073050052941, PRECISION_12);
  }

  @Test
  public void testTrigamma() {
    assertEquals(Approximation.trigamma(1000000), 1.0000005000001667E-6, PRECISION_12);
    assertEquals(Approximation.trigamma(100000), 1.0000050000166667E-5, PRECISION_12);
    assertEquals(Approximation.trigamma(10000), 1.0000500016666666E-4, PRECISION_12);

    assertEquals(Approximation.trigamma(1000), 0.0010005001666666333, PRECISION_12);
    assertEquals(Approximation.trigamma(100), 0.010050166663333571, PRECISION_12);
    assertEquals(Approximation.trigamma(10), 0.10516633568168571, PRECISION_12);

    // precision drops down accordingly when computing digamma function for small value
    assertEquals(Approximation.trigamma(1), 1.6449340668482264, PRECISION_9);
    assertEquals(Approximation.trigamma(0.1), 101.4332991507927, PRECISION_9);
    assertEquals(Approximation.trigamma(0.01), 10001.62121352835, PRECISION_9);
    assertEquals(Approximation.trigamma(0.001), 1000001.6425332422, PRECISION_6);

    assertEquals(Approximation.trigamma(-0.001), 1000001.6473416518, PRECISION_6);
    assertEquals(Approximation.trigamma(-0.01), 10001.669304101055, PRECISION_9);
    assertEquals(Approximation.trigamma(-0.1), 101.92253995947704, PRECISION_9);
  }

  public static junit.framework.Test suite() {
    return new JUnit4TestAdapter(ApproximationTest.class);
  }
}