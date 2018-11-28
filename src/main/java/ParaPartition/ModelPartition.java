package ParaPartition;

import Global.Global;
import ParaStructure.Partitioning.AFMatrix;
import ParaStructure.Partitioning.Partition;
import ParaStructure.Partitioning.PartitionList;
import ParaStructure.Sample.SampleList;
import Util.ListToHashset;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public class ModelPartition {
    public static PartitionList modelPartition(SampleList sampleList,List<Integer> prunedSparseDim,int samplePrunedSize){
        /**
        *@Description: 用来进行模型垂直划分，得到最佳划分
        *@Param: [sampleList, prunedSparseDim, samplePrunedSize]
        *@return: ParaStructure.Partitioning.PartitionList
        *@Author: SongZhen
        *@date: 上午9:07 18-11-28
        */

        PartitionList partitionList= initPartition(prunedSparseDim);
        
        // 遍历数据集建立AF矩阵,返回最佳划分
        PartitionList bestPartitionList=getBestPartition(partitionList,sampleList,prunedSparseDim,samplePrunedSize);
        
        return bestPartitionList;

    }

    public static PartitionList initPartition(List<Integer> prunedSparseDim){
        /**
        *@Description: 初始化每个模型参数到一个划分里
        *@Param: [prunedSparseDim]
        *@return: ParaStructure.Partitioning.PartitionList
        *@Author: SongZhen
        *@date: 上午9:08 18-11-28
        */

        /*初始化partitionList，让稀疏维度的每一维都划分成一个Partition*/
        PartitionList partitionList = new PartitionList();
        for(int i=0;i<prunedSparseDim.size();i++){
            Partition p=new Partition();
            p.partition.add(prunedSparseDim.get(i));
            partitionList.partitionList.add(p);
        }
        return partitionList;
    }


    private static AFMatrix buildAF(PartitionList partitionList, SampleList sampleList, List<Integer> prunedSparseDim,int samplePrunedSize) {
        /**
        *@Description: 建立AF矩阵类，包含af矩阵、当前划分、组合时间成本、时间成本减少值
        *@Param: [partitionList, sampleList, prunedSparseDim, samplePrunedSize]
        *@return: ParaStructure.Partitioning.AFMatrix
        *@Author: SongZhen
        *@date: 上午9:08 18-11-28
        */
        int catSize ;
        int partitionListSize = partitionList.partitionList.size();
        float[][] AF = new float[partitionListSize][partitionListSize];
        AFMatrix afMatrix=new AFMatrix();

        Set<Integer> setPrunedSparseDim=ListToHashset.listToHashSetInt(prunedSparseDim);

        for (int i = 0; i < samplePrunedSize; i++) {  //这是个大循环，在循环所有的数据集

            List<Integer> catNonzeroList = new ArrayList<Integer>();
            catSize=sampleList.sampleList.get(i*(sampleList.sampleListSize/samplePrunedSize)).cat.length;
            int[] cat = sampleList.sampleList.get(i*(sampleList.sampleListSize/samplePrunedSize)).cat;
            for (int j = 0; j < catSize; j++) {  //这个两层循环是遍历所有数据的所有cat维度
                if (cat[j] != -1 && setPrunedSparseDim.contains(cat[j])) { //如果cat的属性不为missing value,且该维度在剪枝后的统计范围内
                    catNonzeroList.add(cat[j]);
                }
            }


            // 如果这一条数据的cat属性能够组合出来Partition，就说明这个partition在这条数据中出现了

            Set<Integer> setCatNonzeroList=ListToHashset.listToHashSetInt(catNonzeroList);
            List<Integer> catContainsPartition=new ArrayList<Integer>();
            for(int l=0;l<partitionListSize;l++){
                Partition partition=partitionList.partitionList.get(l);
                int flag=0;
                for(int m=0;m<partition.partition.size();m++){
                    if(setCatNonzeroList.contains(partition.partition.get(m))){
                        // 这里无论怎么样，都是只包含一个就可以
                        catContainsPartition.add(l);
                        break;

                    }
                }

            }


            for(int l:catContainsPartition){
                for(int m:catContainsPartition){
                    AF[l][m]++;
                }
            }

        }
        afMatrix.AF=AF;
        afMatrix.partitionList=partitionList;
        afMatrix.costTime=new float[partitionList.partitionList.size()][partitionList.partitionList.size()];
        afMatrix.costTimeReduce=new float[partitionList.partitionList.size()][partitionList.partitionList.size()];

        return afMatrix;
    }


    private static PartitionList getBestPartition(PartitionList partitionList, SampleList sampleList, List<Integer> prunedSparseDim, int samplePrunedSize) {
        /**
        *@Description: 获取最佳划分。是一个递归的方式，直到最大时间成本减少值小于等于最小阈值，返回划分结果
        *@Param: [partitionList, sampleList, prunedSparseDim, samplePrunedSize]
        *@return: ParaStructure.Partitioning.PartitionList
        *@Author: SongZhen
        *@date: 上午9:11 18-11-28
        */
        /*先进行第一次计算和合并*/
        float minGain = Global.minGain;
        AFMatrix afMatrix=buildAF(partitionList,sampleList,prunedSparseDim,samplePrunedSize);
        int partitionListSize = afMatrix.partitionList.partitionList.size();


        for (int i = 0; i < partitionListSize; i++) {
            for (int j = 0; j < partitionListSize; j++) {
                if (i == j) {
                    //前面的参数是磁盘访问的两个时间（seek和read），后面是partition[i]包含的Dim个数
                    afMatrix.costTime[i][i] = afMatrix.AF[i][i] * cost(afMatrix.partitionList.partitionList.get(i).partition.size());
                } else {
                    int mergePartitionSize = afMatrix.partitionList.partitionList.get(i).partition.size() + afMatrix.partitionList.partitionList.get(j).partition.size();
                    afMatrix.costTime[i][j] = (afMatrix.AF[i][i] + afMatrix.AF[j][j] - afMatrix.AF[i][j]) * cost(mergePartitionSize);
                }
            }

        }

        float maxTimeReduce = 0;
        int pi = 0;
        int pj = 0;

        /*计算最大的时间成本Reduce，也就是最佳合并pi，pj*/
        for (int i = 0; i < partitionListSize-1; i++) {
            for (int j = i+1; j < partitionListSize; j++) {
                float costReduce = afMatrix.costTime[i][i] + afMatrix.costTime[j][j] - afMatrix.costTime[i][j];
                if (costReduce > maxTimeReduce) {
                    maxTimeReduce = costReduce;
                    pi = i;
                    pj = j;
                }


            }
        }

        /*重新构建partitionList，也就是合并之后的partitionList*/
        if(maxTimeReduce>minGain){
            System.out.println(pi+","+pj);
            int pjSize=getPiSize(afMatrix.partitionList.partitionList,pj);
            for(int i=0;i<pjSize;i++){
              afMatrix.partitionList.partitionList.get(pi).partition.add(afMatrix.partitionList.partitionList.get(pj).partition.get(i));
            }
            afMatrix.partitionList.partitionList.remove(pj);

        }
        else {
            return afMatrix.partitionList;
        }

        return getBestPartition(afMatrix.partitionList,sampleList,prunedSparseDim,samplePrunedSize);

    }




    /*获取划分i的大小*/
    public static int getPiSize(List<Partition> partitionList, int pi){
        return partitionList.get(pi).partition.size();
    }


    /*
     * 计算某个划分Pi的访问时间（seek and read）
     * @para diskAccessTime是磁盘访问的两个时间（seek和read），singlePartitionSize是partition[i]包含的Dim个数
     * @return 返回访问这个Pi的时间成本
     * */

    public static float cost( int singlePartitionSize){
        /**
        *@Description: 这是计算代价损失，表示，如果划分中有一个元素，那么就是按照ParaKV的结构存的，
         * 如果大于1就是按照ParaKVPartition存的。一个ParaKV需要的字节数是70
         * 而一个ParaKVPartition的基础空间是200，每增加一个元素是增加18
        *@Param: [singlePartitionSize]
        *@return: float
        *@Author: SongZhen
        *@date: 上午8:24 18-11-16
        */
        if(singlePartitionSize==1){
            return (Global.diskAccessTime.seekSingleTime+ Global.paraKVSize* Global.diskAccessTime.readSingleDimTime);
        }

        else {
            return (Global.diskAccessTime.seekSingleTime+(singlePartitionSize* Global.singleParaKVOfPartition+ Global.paraKVPartitionBasicSize)* Global.diskAccessTime.readSingleDimTime);
        }

    }








}
