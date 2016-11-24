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



// This pass requires simplifycfg -> loop_rotation passes to be run before it!
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

    bool runOnLoop(Loop *L, LPPassManager &LPM) override {
      LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
      DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
      auto &subLoopsVec = L->getSubLoops();
      if(!((subLoopsVec.size() == 1) && (subLoopsVec[0]->getSubLoops().size() == 0)))
        return false;

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

      // Allocate memory for B loop flag
      Instruction *ti = A_preheader->getTerminator();
      AllocaInst* p_bcond = new AllocaInst(bool_type, "bCondState", ti);
      new StoreInst(False, p_bcond, ti);

      // FIXME: make this work for all possible cases! +case +uncond_jump
      SmallVector<BasicBlock*, 8> ExitingBlocks;
      A->getExitingBlocks(ExitingBlocks);
      for (BasicBlock *ExitingBlock : ExitingBlocks)
        if (BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator()))
          if (BI->isConditional())
          {
            BasicBlock* succ0 = BI->getSuccessor(0);
            Loop* succ_loop0 = LI->getLoopFor(succ0);
            // Store condition flag
            Value* B_condval = BI->getCondition();
            if (succ_loop0 != B)
            {
              BinaryOperator* notB_condval = BinaryOperator::CreateNot(B_condval, "notXcond", BI);
              new StoreInst(notB_condval, p_bcond, BI);
            } 
            else
              new StoreInst(B_condval, p_bcond, BI);
          }

      // Create new intermediate BB for shunting instrs exclusive to A
      BasicBlock* A_shuntkey   = SplitBlock(A_header, A_header->getFirstNonPHI(), DT, LI);
      BasicBlock* A_postheader = SplitBlock(A_shuntkey, A_shuntkey->getFirstNonPHI(), DT, LI);
      // Shunt BBs up to B header
      {
        Instruction *ti = A_shuntkey->getTerminator();
        LoadInst* li = new LoadInst(p_bcond, "condBool", ti);
        ti->eraseFromParent();
        BranchInst::Create(B_header, A_postheader, li, A_shuntkey);
      }
      // Add new empty latch BBs on the A and B's backedges to simplify
      // transformation process
      BasicBlock* A_newlatch = SplitEdge (A_latch, A_header, DT, LI);
      BasicBlock* B_newlatch = SplitEdge (B_latch, B_header, DT, LI);

      // Shunt out everything after B latch to A latch, and reuse A backedge as B backedge
      B_newlatch->getTerminator()->setSuccessor(0, A_newlatch);

      // Store B condition value before jumping back to A header
      Value* B_condval = (cast<BranchInst>(B_latch->getTerminator()))->getCondition();
      new StoreInst(B_condval, p_bcond, B_latch->getTerminator());

      B->invalidate();
      A->invalidate();

      errs() << "LoopF: ";
      ++LoopFCounter;
      return true;
    }
  };
}

char LoopF::ID = 0;
static RegisterPass<LoopF> X("loopf", "Loop Fusion Transform pass by VADER");
