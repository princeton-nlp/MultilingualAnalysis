package maxent;

import edu.jhu.pacaya.gm.maxent.LogLinearXY.LogLinearXYPrm;
import edu.jhu.prim.bimap.IntObjectBimap;
import edu.jhu.prim.vector.IntDoubleVector;
import org.apache.log4j.Logger;
import util.Constant;

import java.io.*;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.zip.GZIPOutputStream;

/**
 * Created by wdd on 14/11/10.
 * Learning model for supervised learning on grammar
 *
 * @author wdd
 */

public class Model implements Iterable<Entry<String, Double>> {

    private static final Logger log = Logger.getLogger(Model.class);
    private CrfLogisticRegression crfLogisticRegression;
    private IntObjectBimap<String> ruleIntObjectBimap, corpusIntObjectBimap, featureIntObjectBimap;
    private Map<String, Double> loadedModelParams;
    private boolean trained;

    @Override
    public Iterator<Entry<String, Double>> iterator() {
        return loadedModelParams.entrySet().iterator();
    }


    public IntObjectBimap<String> getFeatureAlphabet() {
        return featureIntObjectBimap;
    }


    public Model(LogLinearXYPrm prm) throws IOException {
        crfLogisticRegression = new CrfLogisticRegression(prm);
        ruleIntObjectBimap = new IntObjectBimap<>();
        corpusIntObjectBimap = new IntObjectBimap<>();
        featureIntObjectBimap = new IntObjectBimap<>();
        loadedModelParams = new HashMap<>();
        trained = false;
    }

    public void freeze() {
        ruleIntObjectBimap.stopGrowth();
        corpusIntObjectBimap.stopGrowth();
        featureIntObjectBimap.stopGrowth();
    }

    public void saveObj(File to) throws IOException {
        Map<String, Double> modelToSave = new HashMap<>();
        for (String k : loadedModelParams.keySet()) {
            log.info(k + ":" + loadedModelParams.get(k));
            if (loadedModelParams.get(k) != 0.)
                modelToSave.put(k, loadedModelParams.get(k));
        }
        log.info("Save model to: " + to.getAbsolutePath());
        FileOutputStream fos = new FileOutputStream(to);
        GZIPOutputStream gz = new GZIPOutputStream(fos);
        ObjectOutput writer = new ObjectOutputStream(gz);
        writer.writeObject(modelToSave);
        writer.close();
    }

    public void saveText(File to) throws IOException {
        log.info("Save model to: " + to.getAbsolutePath());
        BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(to));
        int tt = 0;
        for (String k : loadedModelParams.keySet()) {
            if (loadedModelParams.get(k) != 0.) {
                tt += 1;
                bufferedWriter.write(k + Constant.FEATURE_VAL_DEL + loadedModelParams.get(k) + "\n");
            }
        }
        log.info(String.format("%d/%d Non-szero parameters", tt, loadedModelParams.size()));
        bufferedWriter.close();
    }

    public void train(DataSet trainData) {
        freeze();
        loadModel();
        crfLogisticRegression.train(trainData);
        saveModel();
        trained = true;
    }

    private void saveModel() {
        loadedModelParams = new HashMap<>();
        IntDoubleVector params = crfLogisticRegression.getParams();
        for (int i = 0; i < params.getNumImplicitEntries(); ++i)
            loadedModelParams.put(featureIntObjectBimap.lookupObject(i), params.get(i));
    }

    private void loadModel() {
        crfLogisticRegression.initModel(featureIntObjectBimap, loadedModelParams);
    }

}
