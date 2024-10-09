<script>
    import Comm from "../Comm.svelte";
</script>

<Comm messages={["m1", "m2", "m3"]} on:receive_message={(e) => console.log("yay", e.detail.message)} />