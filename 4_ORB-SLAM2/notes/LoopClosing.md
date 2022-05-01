# LoopClosing
LoopClosing是专门负责做闭环的类，它的主要功能就是检测闭环，计算闭环帧的相对位姿并以此做闭环修正。

![image1](https://pic2.zhimg.com/80/v2-66d8860e4abbfa5824b59b5dafa6e445_720w.jpg)

## bool LoopClosing::DetectLoop()
言简意赅，返回一个bool类型变量，判断有无闭环；
首先取出了一个关键帧：
```cpp
    {
        // use mutex to extract a key-frame
        unique_lock<mutex> lock(mMutexLoopQueue);  // lock loop queue
        mpCurrentKF = mlpLoopKeyFrameQueue.front();  // extract the first key-frame
        mlpLoopKeyFrameQueue.pop_front();  // pop
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }
```

如果地图包含十个以内的关键帧，则默认没有闭环，返回false：
```cpp
    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    // if(mpCurrentKF->mnID - mLastLoopKFid < 10), easier to understand
    {
        // add to KeyFrame DB, then erase it
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;  // return false, no loop detected
    }
```

接下来是根据BoW模型得到当前帧的`minScore`：
```cpp
    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the co-visibility graph
    // We will impose loop candidates to have a higher similarity than this
    // vpConnectedKeyFrames, pointer that pointing to co-visible KF of mpCurrentKF
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;  // get BoW vector of mpCurrentKF
    float minScore = 1;  // get minScore by using BoW
    // for each KeyFrame that has co-visibility with mpCurrentKF
    // a standard scoring process by BoW below
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        // get BoW vector of KeyFrame in the list of vpConnectedKeyFrames
        const DBoW2::BowVector &BowVec = pKF->mBowVec;
        // get BoW similarity score
        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);
        // minScore is 1, if score by BoW is smaller, then use BoW score
        if(score<minScore)
            minScore = score;
    }
```
从具有共视关系的关键帧中得到score，作为其最小得分















