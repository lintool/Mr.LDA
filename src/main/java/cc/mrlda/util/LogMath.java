package cc.mrlda.util;

public class LogMath {
	/**
	 * 
	 * @param a
	 *            log A, in natural base e
	 * @param b
	 *            log B, in natural base e
	 * @return log(A+B), in natural base e
	 */
	public static double add(double a, double b) {
		if (a < b) {
			return b + Math.log(1 + Math.exp(a - b));
		} else {
			return a + Math.log(1 + Math.exp(b - a));
		}
	}

	/**
	 * 
	 * @param a
	 *            log A, in natural base e
	 * @param b
	 *            log B, in natural base e
	 * @return log(A+B), in natural base e
	 */
	public static float add(float a, float b) {
		if (a < b) {
			return (float) (b + Math.log(1 + Math.exp(a - b)));
		} else {
			return (float) (a + Math.log(1 + Math.exp(b - a)));
		}
	}

	/**
	 * 
	 * @param vector
	 *            a vector in log format of natural base e
	 * @param normalizeFactor
	 *            normalize factor in log format of natural base e
	 */
	public static void normalize(float[] vector, float normalizeFactor) {
		for (int i = 0; i < vector.length; i++) {
			vector[i] -= normalizeFactor;
		}
	}

	/**
	 * 
	 * @param vector
	 *            a vector in log format of natural base e
	 * @param normalizeFactor
	 *            normalize factor in log format of natural base e
	 */
	public static void normalize(double[] vector, double normalizeFactor) {
		for (int i = 0; i < vector.length; i++) {
			vector[i] -= normalizeFactor;
		}
	}

	/**
	 * 
	 * @param vectorA
	 *            a vector A in log format of natural base e
	 * @param vectorB
	 *            a vector B in log format of natural base e
	 * @return
	 */
	public static float[] addVector(float[] vectorA, float[] vectorB) {
		if (vectorA.length != vectorB.length) {
			throw new IllegalArgumentException(
					"Dimension of two vectors does not agree with each other.");
		}

		float[] vectorC = new float[vectorA.length];
		for (int i = 0; i < vectorA.length; i++) {
			vectorC[i] = add(vectorA[i], vectorB[i]);
		}

		return vectorC;
	}

	/**
	 * 
	 * @param vectorA
	 *            a vector A in log format of natural base e
	 * @param vectorB
	 *            a vector B in log format of natural base e
	 */
	public static void addVectorToA(float[] vectorA, float[] vectorB) {
		if (vectorA.length != vectorB.length) {
			throw new IllegalArgumentException(
					"Dimension of two vectors does not agree with each other.");
		}

		for (int i = 0; i < vectorA.length; i++) {
			vectorA[i] = add(vectorA[i], vectorB[i]);
		}
	}
}