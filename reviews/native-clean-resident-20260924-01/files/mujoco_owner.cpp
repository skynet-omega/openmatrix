#include "mujoco_owner.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace epoch {

MuJoCoBodyOwner::MuJoCoBodyOwner(mjModel* model, mjData* live, std::string motor_port)
    : model_(model), live_(live), backup_(nullptr), motor_port_(std::move(motor_port)) {
  if (!model_ || !live_ || motor_port_.empty()) throw std::runtime_error("invalid MuJoCo owner");
  if (model_->nu != 1) throw std::runtime_error("fixture body owner requires one actuator");
  backup_ = mj_makeData(model_);
  if (!backup_) throw std::runtime_error("cannot allocate MuJoCo rollback data");
}

MuJoCoBodyOwner::~MuJoCoBodyOwner() { mj_deleteData(backup_); }

void MuJoCoBodyOwner::snapshot() {
  if (!mj_copyData(backup_, model_, live_)) throw std::runtime_error("MuJoCo snapshot failed");
  has_snapshot_ = true;
}

void MuJoCoBodyOwner::advance(Context& context) {
  live_->ctrl[0] = context.value(motor_port_);
  mj_step(model_, live_);
  if (!std::isfinite(live_->time) || !std::isfinite(live_->qpos[0])) {
    throw std::runtime_error("nonfinite MuJoCo state");
  }
}

void MuJoCoBodyOwner::commit() {}

void MuJoCoBodyOwner::rollback() noexcept {
  if (has_snapshot_) mj_copyData(live_, model_, backup_);
}

void MuJoCoBodyOwner::release_snapshot() noexcept { has_snapshot_ = false; }

}  // namespace epoch
