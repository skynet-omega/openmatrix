#pragma once

#include "epoch_session.hpp"

#include <mujoco/mujoco.h>

namespace epoch {

// The caller owns model and live data. This owner holds only its rollback copy.
class MuJoCoBodyOwner final : public Owner {
 public:
  MuJoCoBodyOwner(mjModel* model, mjData* live, std::string motor_port);
  ~MuJoCoBodyOwner() override;
  std::string name() const override { return "mujoco_body"; }
  std::vector<Port> reads() const override { return {{motor_port_, "motor_ctrl"}}; }
  std::vector<Port> writes() const override { return {}; }
  void snapshot() override;
  void advance(Context& context) override;
  void commit() override;
  void rollback() noexcept override;
  void release_snapshot() noexcept override;

 private:
  mjModel* model_;
  mjData* live_;
  mjData* backup_;
  std::string motor_port_;
  bool has_snapshot_ = false;
};

}  // namespace epoch
