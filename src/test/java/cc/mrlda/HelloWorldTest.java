package cc.mrlda;

import static org.junit.Assert.assertEquals;
import junit.framework.JUnit4TestAdapter;

import org.junit.Test;

public class HelloWorldTest {

  @Test
  public void test() throws Exception {
    assertEquals(1, 1);
  }

  public static junit.framework.Test suite() {
    return new JUnit4TestAdapter(HelloWorldTest.class);
  }
}
