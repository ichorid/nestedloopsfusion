#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

#define DEBUG_TYPE "loopf"

STATISTIC(LoopFCounter, "Loop Fusion Transform counter :");

namespace {
  struct LoopF : public LoopPass{
    static char ID; // Pass identification, replacement for typeid
    LoopF() : LoopPass(ID) {}

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<LoopInfoWrapperPass>();
    }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override {
      LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      auto &subLoopsVec = L->getSubLoops();
      if((subLoopsVec.size() == 1) && (subLoopsVec[0]->getSubLoops().size() == 0)){
        LLVMContext &Context = L->getHeader()->getContext();
        Value *False = ConstantInt::getFalse(Context);
        IntegerType *bool_type = Type::getInt1Ty(Context);
        Loop *A = L;
        Loop *B = L->getSubLoops()[0];
        BasicBlock *A_header = A->getHeader();
        BasicBlock *B_header = B->getHeader();
        BasicBlock *A_latch = A->getLoopLatch();
        BasicBlock *B_latch = B->getLoopLatch();
        BasicBlock *A_preheader = A->getLoopPreheader();
        Instruction *pi = A_preheader->getTerminator(); 
        AllocaInst* ai = new AllocaInst(bool_type, "bCondState", pi);
        StoreInst* si = new StoreInst(False, ai, pi);
        errs() << "LoopF: ";
        ++LoopFCounter;
        return true;
      }
      return false;
    }
  };
}

char LoopF::ID = 0;
static RegisterPass<LoopF> X("loopf", "Loop Fusion Transform pass by VADER");
