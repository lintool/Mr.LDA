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
import junit.framework.JUnit4TestAdapter;

import org.junit.Test;

public class VariationalInferenceTest {
  public static double PRECISION_10 = 1e-10;

  @Test
  public void testUpdateAlphaVector() {
    double[] alphaVector = { 0.4736839726180464, 9.928726975283879, 8.319361678447014 };
    double[] alphaSufficientStatistics = { -23792.9569126969113, -22519.9434073184025,
        -23973.2360888324797 };
    double[] alphaUpdateVector = VariationalInference.updateVectorAlpha(3, 112, alphaVector,
        alphaSufficientStatistics);
  }

  @Test
  public void testUpdateAlphaScalar() {
    assertEquals(VariationalInference.updateScalarAlpha(5, 2246, 100, -40100.9192398908126052),
        0.2958548131184747, PRECISION_10);
    assertEquals(VariationalInference.updateScalarAlpha(5, 2246, 100, -34828.2371112336259102),
        0.3731832583179411, PRECISION_10);
    assertEquals(VariationalInference.updateScalarAlpha(5, 2246, 100, -37309.1699276268700487),
        0.3319329678764105, PRECISION_10);
    assertEquals(VariationalInference.updateScalarAlpha(5, 2246, 100, -44085.8660385293114814),
        0.2568195157403902, PRECISION_10);

    assertEquals(VariationalInference.updateScalarAlpha(10, 2246, 100, -155990.5727383689954877),
        0.1531475153565107, PRECISION_10);
    assertEquals(VariationalInference.updateScalarAlpha(10, 2246, 100, -196359.2521305996051524),
        0.1150183709445565, PRECISION_10);
    assertEquals(VariationalInference.updateScalarAlpha(10, 2246, 100, -226577.3570433593704365),
        0.0972395316113154, PRECISION_10);
    assertEquals(VariationalInference.updateScalarAlpha(10, 2246, 100, -256318.9209672076685820),
        0.0845206104885002, PRECISION_10);
  }

  public static junit.framework.Test suite() {
    return new JUnit4TestAdapter(VariationalInferenceTest.class);
  }
}