import Global.Global;
import KVStore.ParaKVStore;
import ParaPartition.ModelPartition;
import ParaStructure.KVPara.CatParaList;
import ParaStructure.KVPara.FeatureParaList;
import ParaStructure.KVPara.InvertIndex;
import ParaStructure.Partitioning.PartitionList;
import ParaStructure.Sample.SampleList;
import Util.*;
import org.iq80.leveldb.DB;

import java.io.IOException;
import java.util.List;


public class LogisticRegression {

    public static int sparseDimSize=0;
    public static int featureSize= Global.featureSize;


    public static void main(String[] args) throws IOException,ClassNotFoundException {

        String fileName= Global.dataFilePath;
        int batchSize= Global.batchSize;
        SampleList sampleList;

        // 定义feature和cat大小

        int catSize= Global.catSize;



        // 测试ParaKV和包含一个ParaKV的ParaKVList的大小
        TestUtil.spaceCostTest();


        // 处理并读取训练集数据到sampleList中
        CurrentTimeUtil.setStartTime();
        sampleList=DataProcessUtil.readData(fileName,featureSize,catSize);
        CurrentTimeUtil.setEndTime();
        CurrentTimeUtil.showExecuteTime("readDataTime:");

        // 将训练集的连续特征标准化
        CurrentTimeUtil.setStartTime();
        sampleList=DataProcessUtil.linerNormalization(sampleList);
        CurrentTimeUtil.setEndTime();
        CurrentTimeUtil.showExecuteTime("normalizationTime:");

        // sparseDimensionSize
        sparseDimSize=sampleList.sparseDimSize;
        sampleList.showSparseDimSize();

        // 不对参数进行垂直划分，直接读
        readWithoutPartition(sampleList);


        // 论文中time测试，包含了bestPartitionList和最笨的测试
        reimplTimeTest(sampleList);


        // batch-based划分方法
        batchBasedMethodTest(sampleList,batchSize);





    }



    public static void reimplTimeTest(SampleList sampleList) throws IOException,ClassNotFoundException{

        /**
        *@Description: 实现论文中的方法，也就是针对SGD的，一条数据一条数据读
        *@Param: [sampleList]
        *@return: void
        *@Author: SongZhen
        *@date: 下午7:50 18-11-12
        */
        // 初始化模型,包括kvStore，离散和连续属性对应的参数列表
        ParaKVStore paraKVStore=new ParaKVStore();
        CatParaList catParaList=new CatParaList();
        FeatureParaList featureParaList=new FeatureParaList();

        // 维度剪枝
        float pruneRate= Global.pruneRate;
        List<Integer> prunedSparseDim=PruneUtil.prune(sampleList,pruneRate, Global.freqThresholdForSingle, Global.singleSamplePrunedSize);
        System.out.println("prunedSparseDim:"+prunedSparseDim.size());

        // 模型划分，得到最佳划分
        CurrentTimeUtil.setStartTime();
        PartitionList bestPartitionList= ModelPartition.modelPartition(sampleList,prunedSparseDim, Global.singleSamplePrunedSize);

        // 划分中有一些划分只包含一个维度，这样的维度定义成single ParaKV就行了，不用定义成list。
        prunedSparseDim=PruneUtil.removeOnceItemOfBPL(bestPartitionList);
        bestPartitionList=PartitionList.getBestPartitionListWithNoOnceItem(bestPartitionList);

        // 显示bestPartitionList
        bestPartitionList.showPartitionList();

        CurrentTimeUtil.setEndTime();
        CurrentTimeUtil.showExecuteTime("SGDPartitionTime:");


        // 初始化离散和连续特征对应的参数列表（离散属性已对应上最佳划分）
        featureParaList.featureParaList=ParaKVStore.initFeaturePara(featureSize);
        catParaList.catParaList=ParaKVStore.initCatPara(bestPartitionList);


        DB db=ParaKVStore.initParaKVstore(featureParaList,catParaList,prunedSparseDim, Global.dbForSGDFilePath,sampleList.sparseDimSize);
        paraKVStore.db=db;


        // bestPartitionList,遍历所有的样本，一个一个样本的读取所有数据
        List<InvertIndex> invertIndices=AccessUtil.getInvertIndices(sampleList,bestPartitionList);
        CurrentTimeUtil.setStartTime();
        AccessUtil.accessParaByInvertIndices(invertIndices,paraKVStore);
        CurrentTimeUtil.setEndTime();
        CurrentTimeUtil.showExecuteTime("SGDReadTime:");


        db.close();
    }


    public static void readWithoutPartition(SampleList sampleList) throws IOException,ClassNotFoundException{
        /*构建一个大小sparseDimSize的参数模型，然后一条数据一条数据读*/
        ParaKVStore paraKVStore=new ParaKVStore();
        DB db=ParaKVStore.buildLevelDB(Global.dbForNoVerticalPartitionPath);
        paraKVStore.db=db;
        AccessUtil.initPara(sparseDimSize,db);
        CurrentTimeUtil.setStartTime();
        AccessUtil.getPara(sampleList,paraKVStore);
        CurrentTimeUtil.setEndTime();
        CurrentTimeUtil.showExecuteTime("singleReadTime：");

        db.close();
    }



    public static void batchBasedMethodTest(SampleList sampleList, int batchSize)throws IOException,ClassNotFoundException{
        ParaKVStore paraKVStore=new ParaKVStore();
        CatParaList catParaList=new CatParaList();
        FeatureParaList featureParaList=new FeatureParaList();

        // 这个partitionList是每个batch需要访问的维度
        PartitionList batchNeedAccessList=BatchUtil.getBatchNeedPara(sampleList,batchSize);
        SampleList batchList= BatchUtil.batchToSampleList(batchNeedAccessList,sampleList);

        // 维度剪枝
        float pruneRate= Global.pruneRate;
        List<Integer> prunedSparseDim=PruneUtil.prune(batchList,pruneRate, Global.freqThresholdForBatch, Global.batchSamplePrunedSize);
        System.out.println("prunedDimSize:"+prunedSparseDim.size());

        CurrentTimeUtil.setStartTime();
        PartitionList bestPartitionList=ModelPartition.modelPartition(batchList,prunedSparseDim, Global.batchSamplePrunedSize);



        // 下面两行代码分别是从剪枝后的维度列表和最佳划分中，除去只包含一个维度的划分
        prunedSparseDim=PruneUtil.removeOnceItemOfBPL(bestPartitionList);
        bestPartitionList=PartitionList.getBestPartitionListWithNoOnceItem(bestPartitionList);

        // 显示bestPartitionList
        bestPartitionList.showPartitionList();

        CurrentTimeUtil.setEndTime();
        CurrentTimeUtil.showExecuteTime("batchPartitionTime:");


        featureParaList.featureParaList=ParaKVStore.initFeaturePara(featureSize);
        catParaList.catParaList=ParaKVStore.initCatPara(bestPartitionList);



        DB db=ParaKVStore.initParaKVstore(featureParaList,catParaList,prunedSparseDim, Global.dbForMiniBGDFilePath,sparseDimSize);
        paraKVStore.db=db;

        List<InvertIndex> invertIndices=AccessUtil.getInvertIndices(batchList,bestPartitionList);

        CurrentTimeUtil.setStartTime();
        AccessUtil.accessParaByInvertIndices(invertIndices,paraKVStore);
        CurrentTimeUtil.setEndTime();
        CurrentTimeUtil.showExecuteTime("Batch-based Read:");

        db.close();
    }




}
