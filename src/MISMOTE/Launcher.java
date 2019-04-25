/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package MISMOTE;

import weka.core.Instances;
import weka.filters.Filter;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.ConverterUtils.DataSink;

/**
 *
 * @author Danel
 */
public class Launcher {

  /**
   * Configuración del problema WIR original (hold-out)
   */
  //final static private String genericPath = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/WebMIL/less-word/tfc-tfc/";
  //final static private String fileName = "MI-tfc-tfc-V%d-words5-40.%s.arff"; // %d = 1..9, %s = test, train
  /**
   * Configuración del problema WIR preparado para 5x5CV
   */
  final static private String genericPath = "C:/Users/Danel/Documents/Investigación/LAB/Datasets/multiInstance/IMBALANCED/5CV/wir-v%d/";
  final static private String fileName = "wir-v%d-r%d-f%d-%s.arff"; // %d = 1..9, %s = test, train

  public static void doMismote_5x5CV() {
    try {
      for (int i = 5; i <= 5; i++) {
        String specificPath = String.format(genericPath, i);
        for (int run = 1; run <= 5; run++) {
          for (int fold = 1; fold <= 5; fold++) {
            String trainName = String.format(fileName, i, run, fold, "train");
            Instances trainData = DataSource.read(specificPath + trainName);
            trainData.setClassIndex(trainData.numAttributes() - 1);
            System.out.println("MISMOTing " + trainName);
            MISMOTE mismote = new MISMOTE();
            mismote.setInputFormat(trainData);
            Instances trainInstances = Filter.useFilter(trainData, mismote);
            String outputFile = String.format("%s/Mismo_%s", specificPath, trainName);
            DataSink.write(outputFile, trainInstances);
          }
        }
      }
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  public static void doMismote_HoldOut() {
    try {
      for (int i = 1; i <= 3; i++) {
        String trainName = String.format(fileName, i, "train");
        Instances trainData = DataSource.read(genericPath + trainName);
        trainData.setClassIndex(trainData.numAttributes()-1);
        MISMOTE mismote = new MISMOTE();
        mismote.setInputFormat(trainData);
        Instances trainInstances = Filter.useFilter(trainData, mismote);
        String outputFile = String.format("%s/Mismo_%s", genericPath, trainName);
        DataSink.write(outputFile, trainInstances);
      }
    } catch (Exception e) {
         System.err.println(e.getMessage());
    }
  }

  /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
      doMismote_5x5CV();
    }

}
