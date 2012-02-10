package cc.mrlda.util;

import com.google.common.base.Preconditions;

public class Approximation {
  /**
   * Approximate digamma of x.
   * 
   * @param x
   * @return
   */
  public static double digamma(double x) {
    double r = 0.0;

    while (x <= 5) {
      r -= 1 / x;
      x += 1;
    }

    double f = 1.0 / (x * x);
    // double t = f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0 + f
    // * (-1.0 / 132.0 + f * (691.0 / 32760.0 + f * (-1.0 / 12.0 + f * 3617.0 / 8160.0)))))));
    double t = f
        * (-0.0833333333333333333333333333333 + f
            * (0.00833333333333333333333333333333 + f
                * (-0.00396825396825396825 + f
                    * (0.0041666666666666666666666667 + f
                        * (-0.00757575757575757575757575757576 + f
                            * (0.0210927960928 + f
                                * (-0.0833333333333333333333333333333 + f * 0.44325980392157)))))));
    return r + Math.log(x) - 0.5 / x + t;
  }

  /**
   * Approximate the trigamma of x.
   * 
   * @param x
   * @return
   */
  public static double trigamma(double x) {
    double p;
    int i;

    x = x + 6;
    p = 1 / (x * x);
    p = (((((0.075757575757576 * p - 0.033333333333333) * p + 0.0238095238095238) * p - 0.033333333333333)
        * p + 0.166666666666667)
        * p + 1)
        / x + 0.5 * p;
    for (i = 0; i < 6; i++) {
      x = x - 1;
      p = 1 / (x * x) + p;
    }

    Preconditions.checkArgument(!Double.isNaN(p), new ArithmeticException(
        "invalid input at trigamma function: " + x));

    if (Double.isNaN(p)) {
      throw new ArithmeticException("invalid input at trigamma function: " + x);
    }

    return p;
  }
}