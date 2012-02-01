package cc.mrlda.util;

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
    double t = f
        * (-1.0 / 12.0 + f
            * (1.0 / 120.0 + f
                * (-1.0 / 252.0 + f
                    * (1.0 / 240.0 + f
                        * (-1.0 / 132.0 + f
                            * (691.0 / 32760.0 + f * (-1.0 / 12.0 + f * 3617.0 / 8160.0)))))));
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

    if (new Double(p).equals(Double.NaN)) {
      throw new ArithmeticException("invalid input at trigamma function: " + x);
    }

    return p;
  }

  /**
   * Approximate the lngamma of x.
   * 
   * @param x
   * @return
   */
  public static double lnGamma(double x) {
    double z = 1 / (x * x);

    x = x + 6;
    z = (((-0.000595238095238 * z + 0.000793650793651) * z - 0.002777777777778) * z + 0.083333333333333)
        / x;
    z = (x - 0.5) * Math.log(x) - x + 0.918938533204673 + z - Math.log(x - 1) - Math.log(x - 2)
        - Math.log(x - 3) - Math.log(x - 4) - Math.log(x - 5) - Math.log(x - 6);

    if (new Double(z).equals(Double.NaN)) {
      throw new ArithmeticException("invalid input at lnGamma function: " + x);
    }

    return z;
  }
}
