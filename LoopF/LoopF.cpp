#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/IR/InstrTypes.h"


using namespace llvm;

#define DEBUG_TYPE "loopf"

STATISTIC(LoopFCounter, "Loop Fusion Transform counter :");

namespace {
  struct LoopF : public LoopPass{
    static char ID; // Pass identification, replacement for typeid
    LoopF() : LoopPass(ID) {}

    LoopInfo *LI;
    DominatorTree *DT;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      //AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<DominatorTreeWrapperPass>();
      getLoopAnalysisUsage(AU);
    }

    BasicBlock* shuntRegion(BasicBlock* Head, BasicBlock* Bottom, AllocaInst* condBool)
    {
      // FIXME SplitEdge behaves in strange ways...
      BasicBlock* jumpBB = SplitEdge(Head, Head->getSingleSuccessor(), DT, LI);
      jumpBB = jumpBB->getSinglePredecessor();
      Instruction *pi = jumpBB->getTerminator();
      LoadInst* li = new LoadInst(condBool, "condBool", pi);
      BasicBlock* HeaderSuccessor = jumpBB->getSingleSuccessor();
      jumpBB->getTerminator()->eraseFromParent();
      BranchInst::Create(Bottom, HeaderSuccessor, li, jumpBB);
      return jumpBB;
    }

    int GetBackedgeNum (Loop *L){
      BasicBlock *latch = L->getLoopLatch();
      BasicBlock *header = L->getHeader();
      BasicBlock* Succ0 = cast<BranchInst>(latch->getTerminator())->getSuccessor(0);
      BasicBlock* Succ1 = cast<BranchInst>(latch->getTerminator())->getSuccessor(1);
      int BackedgeNum = ((header == Succ0) ? 0 : 1);
      return BackedgeNum;
    }
    bool runOnLoop(Loop *L, LPPassManager &LPM) override {
      LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
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
        const int A_BackedgeNum = GetBackedgeNum(A);
        const int B_BackedgeNum = GetBackedgeNum(B);

        Instruction *pi = A_preheader->getTerminator();
        AllocaInst* ai = new AllocaInst(bool_type, "bCondState", pi);
        new StoreInst(False, ai, pi);
        AllocaInst* p_acond = new AllocaInst(bool_type, "aCondState", pi);
        new StoreInst(False, p_acond, pi);


        // Create new header and latch for A loop
        BasicBlock* A_newheader = SplitEdge(A_preheader, A_header, DT, LI);
        Value* A_condval = cast<BranchInst>(A_latch->getTerminator())->getCondition();
        new StoreInst(A_condval, p_acond, A_latch->getTerminator());

        BasicBlock* A_newlatch = SplitBlock(A_latch, A_latch->getTerminator(), DT, LI);
        A_newlatch->setName("A.newlatch");
        A_newlatch->getTerminator()->setSuccessor(A_BackedgeNum, A_newheader);


        // Predicate out everything starting from [A header up to B header)
        BasicBlock* A_headershuntlatch = shuntRegion (A_newheader, B_header, ai);

        // Store B condition value before jumping back to A header
        Value* B_condval = cast<BranchInst>(B_latch->getTerminator())->getCondition();
        new StoreInst(B_condval, ai, B_latch->getTerminator());
        

        // Load conditions
        LoadInst* A_cond_loaded = new LoadInst(p_acond, "AcondBool", A_newlatch->getTerminator());
        LoadInst* B_cond_loaded = new LoadInst(ai, "BcondBool", A_newlatch->getTerminator());
        BinaryOperator* orAB = BinaryOperator::Create(Instruction::Or, A_cond_loaded, B_cond_loaded, "or.ab", A_newlatch->getTerminator());
        cast<BranchInst>(A_newlatch->getTerminator())->setCondition(orAB);

        // Predicate out everything starting from (B_latch up to A_newlatch)
        // Due to a bug in SplitEdge we use SplitBlock instead
        BasicBlock* A_newprelatch = SplitBlock(A_newlatch->getSinglePredecessor(), A_newlatch->getSinglePredecessor()->getTerminator(), DT, LI);
        B_latch->getTerminator()->setSuccessor(B_BackedgeNum, A_newprelatch);

        B->invalidate();
        A->invalidate();

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
