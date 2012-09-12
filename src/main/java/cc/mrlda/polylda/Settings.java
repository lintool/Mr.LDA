package cc.mrlda.polylda;

public interface Settings extends cc.mrlda.Settings {
  static final String PROPERTY_PREFIX = Settings.class.getPackage().getName() + "" + DOT;

  public static final String LANGUAGE_OPTION = "language";

  public static final String LANGUAGE_INDICATOR = "lang";
}